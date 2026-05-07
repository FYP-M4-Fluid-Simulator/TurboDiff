"""XFoil runner and DAT file writer for TurboDiff.

Provides:
  - run_xfoil(dat_file, re, aoa) → (Cl, Cd) | None
  - write_dat_file(wu, wl, filepath, ...)

These utilities are used both by the example scripts (optimize_naca_suite.py,
optimize_airfoil_rans.py) and by the FastAPI optimization server to validate
candidate airfoil shapes against XFoil polar data.

XFoil path resolution
---------------------
The binary is resolved in this priority order:
  1. Environment variable  XFOIL_PATH
  2. Common Mac install    /usr/local/bin/xfoil
  3. Homebrew arm64        /opt/homebrew/bin/xfoil
  4. Custom Mac build      /Users/musab/Xfoil-for-Mac/bin/xfoil  (fallback)

Set XFOIL_PATH to override for your environment.
"""

from __future__ import annotations

import os
import subprocess
import shutil
import logging
from typing import Tuple, Optional

import numpy as np
import jax.numpy as jnp

from turbodiff.core.airfoil import generate_cst_coords

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XFoil binary resolution
# ---------------------------------------------------------------------------

_FALLBACK_PATHS = [
    "/usr/bin/xfoil",  # Standard Linux (Apt)
    "/usr/local/bin/xfoil",
    "/opt/homebrew/bin/xfoil",
    "/Users/musab/Xfoil-for-Mac/bin/xfoil",
]


def _resolve_xfoil() -> str:
    """Return the first usable XFoil binary path, or '' if none found."""
    # 1. Explicit env var (set XFOIL_PATH in .env)
    env_path = os.environ.get("XFOIL_PATH", "")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    # 2. System PATH
    found = shutil.which("xfoil")
    if found:
        return found
    # 3. Hard-coded Mac fallbacks
    for p in _FALLBACK_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return ""


XFOIL_PATH: str = _resolve_xfoil()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_xfoil(
    dat_file: str,
    re: float,
    aoa: float,
    xfoil_path: str | None = None,
    timeout: int = 20,  # Match example script timeout
    max_iter: int = 300,
) -> Optional[Tuple[float, float]]:
    """Run XFoil on a Selig-format .dat file and return (Cl, Cd) or None."""
    binary = xfoil_path or XFOIL_PATH
    if not binary:
        logger.error("XFoil binary not found.")
        return None

    print(f"Running XFoil for {os.path.basename(dat_file)} with Re={re} and AoA={aoa}")

    # Use bare filenames — XFoil on Mac silently fails to open/create files
    # when given absolute paths in LOAD / PACC commands. Running with cwd set
    # to the dat directory and passing only the filename reliably produces the polar.
    work_dir = os.path.dirname(os.path.abspath(dat_file))
    dat_name = os.path.basename(dat_file)
    polar_name = dat_name.replace(".dat", ".txt")
    polar_file = os.path.join(work_dir, polar_name)

    if os.path.exists(polar_file):
        os.remove(polar_file)

    # Strictly following the triple-quoted command structure for reliability
    commands = f"""LOAD {dat_name}
PANE
OPER
ITER {max_iter}
VISC {re}
PACC
{polar_name}

ALFA {aoa}

QUIT
"""

    try:
        env = os.environ.copy()
        env["DISPLAY"] = ":0"
        process = subprocess.Popen(
            binary,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=work_dir,  # ← run from the dat file's directory
        )
        stdout, stderr = process.communicate(input=commands, timeout=timeout)

        # XFoil on Mac/gfortran frequently exits with a non-zero code due to a
        # Fortran EOF error triggered when stdin closes. This is cosmetic.
        if process.returncode != 0:
            err_msg = stderr.strip() if stderr else ""
            if "EOF" not in err_msg:
                logger.warning(
                    f"XFoil exited with code {process.returncode}. stderr: {err_msg[:200]}"
                )

    except subprocess.TimeoutExpired:
        logger.error(f"XFoil timed out after {timeout}s for {dat_file}")
        return None
    except Exception as e:
        logger.error(f"Exception running XFoil: {e}")
        return None

    if not os.path.exists(polar_file):
        logger.warning(f"XFoil did not produce a polar file for {dat_file}. stdout:\n{stdout[:500]}")
        return None

    try:
        with open(polar_file, "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # Ensure first 3 parts are valid floats (AoA, Cl, Cd)
                        p0, p1, p2 = float(parts[0]), float(parts[1]), float(parts[2])
                        if abs(p0 - aoa) < 0.01:
                            return p1, p2
                    except ValueError:
                        continue  # Skip header/text lines

        logger.warning(f"XFoil converged but AoA={aoa} not found in {polar_file}")
    except Exception as e:
        logger.error(f"Error parsing XFoil polar file: {e}")
        pass

    return None


def write_dat_file(
    weights_upper: np.ndarray | jnp.ndarray,
    weights_lower: np.ndarray | jnp.ndarray,
    filepath: str,
    label: str = "CST Airfoil",
    num_points: int = 200,
) -> None:
    """Write a Selig-format .dat file for the CST airfoil defined by the weights."""
    wu_j = jnp.array(weights_upper)
    wl_j = jnp.array(weights_lower)
    x_norm, y_upper, y_lower = generate_cst_coords(wu_j, wl_j, num_points=num_points)
    x_np = np.array(x_norm)
    yu = np.array(y_upper)
    yl = np.array(y_lower)

    with open(filepath, "w") as f:
        f.write(f"{label}\n")
        # Upper surface: TE → LE
        for xi, yi in zip(x_np[::-1], yu[::-1]):
            f.write(f"  {xi:.6f}  {yi:.6f}\n")
        # Lower surface: LE → TE
        for xi, yi in zip(x_np, yl):
            f.write(f"  {xi:.6f}  {yi:.6f}\n")
