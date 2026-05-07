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
import re as re_mod
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
    """Run XFoil on a Selig-format .dat file and return (Cl, Cd) or None.

    Uses the same subprocess pattern as test_xfoil.py (which is confirmed
    working inside the Docker container):
      - Invoke xfoil as a list arg to Popen
      - No env / DISPLAY manipulation
      - Parse CL and CD from stdout using regex (no polar file)
    """
    binary = xfoil_path or XFOIL_PATH
    if not binary:
        logger.error("XFoil binary not found.")
        return None

    dat_name = os.path.basename(dat_file)
    work_dir = os.path.dirname(os.path.abspath(dat_file))

    print(f"Running XFoil for {dat_name} with Re={re} and AoA={aoa}")

    # Command sequence mirrors the working test_xfoil.py pattern.
    # LOAD + PANE replaces NACA since we have a custom .dat file.
    commands = f"""LOAD {dat_name}
PANE
OPER
ITER {max_iter}
VISC {re}
ALFA {aoa}
QUIT
"""

    try:
        process = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=work_dir,
        )
        stdout, stderr = process.communicate(input=commands, timeout=timeout)

        if process.returncode != 0:
            err_msg = stderr.strip() if stderr else ""
            if "EOF" not in err_msg:
                logger.warning(
                    "XFoil exited with code %s. stderr: %s\nstdout:\n%s",
                    process.returncode, err_msg[:200], stdout[:800],
                )

    except subprocess.TimeoutExpired:
        logger.error(f"XFoil timed out after {timeout}s for {dat_file}")
        process.kill()
        process.wait()
        return None
    except FileNotFoundError:
        logger.error("'xfoil' command not found. Ensure it is installed and in PATH.")
        return None
    except Exception as e:
        logger.error(f"Exception running XFoil: {e}")
        return None

    # Parse CL and CD from stdout — same regex approach as test_xfoil.py
    cl_matches = re_mod.findall(r"CL\s*=\s*([+-]?\d*\.?\d+)", stdout)
    cd_matches = re_mod.findall(r"CD\s*=\s*([+-]?\d*\.?\d+)", stdout)

    if cl_matches and cd_matches:
        cl = float(cl_matches[-1])
        cd = float(cd_matches[-1])
        logger.info("XFoil converged: Cl=%s Cd=%s", cl, cd)
        print(f"XFoil converged: Cl={cl}, Cd={cd}")
        return cl, cd
    else:
        logger.warning(
            "XFoil did not converge for Re=%.2e, AoA=%.1f. stdout:\n%s",
            re, aoa, stdout[:500],
        )
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
