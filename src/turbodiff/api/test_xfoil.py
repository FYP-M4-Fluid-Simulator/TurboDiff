"""XFoil runner and DAT file writer for TurboDiff.

Provides:
  - run_xfoil(dat_file, re, aoa) → (Cl, Cd) | None
  - write_dat_file(wu, wl, filepath, ...)
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

def _resolve_xfoil() -> str:
    """Return the first usable XFoil binary path, or '' if none found."""
    env_path = os.environ.get("XFOIL_PATH", "")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    
    found = shutil.which("xfoil")
    if found:
        return found
    
    # Mac fallbacks just in case
    for p in ["/usr/local/bin/xfoil", "/opt/homebrew/bin/xfoil"]:
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
    timeout: int = 20,
    max_iter: int = 300,
) -> Optional[Tuple[float, float]]:
    """Run XFoil on a Selig-format .dat file and return (Cl, Cd) or None."""
    binary = xfoil_path or XFOIL_PATH
    if not binary:
        logger.error("XFoil binary not found.")
        return None

    print(f"Running XFoil for {os.path.basename(dat_file)} with Re={re} and AoA={aoa}")

    work_dir = os.path.dirname(os.path.abspath(dat_file))
    dat_name = os.path.basename(dat_file)
    polar_name = dat_name.replace(".dat", ".txt")
    polar_file = os.path.join(work_dir, polar_name)

    if os.path.exists(polar_file):
        os.remove(polar_file)

    # 1. Bulletproof Headless Command List
    commands_list = [
        "PLOP",
        "G F",              # Turn off X11 Graphics (Critical for Docker)
        "",                 # Exit PLOP menu
        f"LOAD {dat_name}", # Load Airfoil
        "PANE",             # Smooth panelling
        "OPER",
        f"ALFA {aoa}",      # Inviscid initialization (prevents math crashes)
        f"VISC {re}",       # Turn on viscous mode
        f"ITER {max_iter}",
        "PACC",             # Start recording Polar
        polar_name,         # Provide save filename
        "",                 # Provide empty string to skip the dump filename
        f"ALFA {aoa}",      # Run the viscous simulation
        "",                 # Exit OPER menu
        "QUIT"              # Safely exit XFOIL
    ]
    
    commands = "\n".join(commands_list) + "\n"

    try:
        # 2. Run completely isolated from display variables
        process = subprocess.run(
            [binary],
            input=commands,
            text=True,
            capture_output=True,
            cwd=work_dir,
            timeout=timeout,
        )

        # Catch specific Fortran math crashes
        if process.returncode == -8:
            logger.warning(f"XFOIL SIGFPE Math Crash for {dat_name} at AoA={aoa}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"XFoil timed out after {timeout}s for {dat_file}")
        return None
    except Exception as e:
        logger.error(f"Exception running XFoil: {e}")
        return None

    if not os.path.exists(polar_file):
        return None

    # 3. Extract the polar data
    try:
        with open(polar_file, "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        p0, p1, p2 = float(parts[0]), float(parts[1]), float(parts[2])
                        if abs(p0 - aoa) < 0.01:
                            return p1, p2
                    except ValueError:
                        continue

        logger.warning(f"XFoil converged but AoA={aoa} not found in {polar_file}")
    except Exception as e:
        logger.error(f"Error parsing XFoil polar file: {e}")

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
        # Upper surface: TE -> LE
        for xi, yi in zip(x_np[::-1], yu[::-1]):
            f.write(f"  {xi:.6f}  {yi:.6f}\n")
        # Lower surface: LE -> TE
        for xi, yi in zip(x_np, yl):
            f.write(f"  {xi:.6f}  {yi:.6f}\n")


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # NACA 0012 Approximate CST Weights
    NACA0012_W_U = jnp.array([0.1726, 0.1269, 0.1441, 0.1080, 0.1750, 0.1471])
    NACA0012_W_L = -NACA0012_W_U

    dat_file = os.path.join(current_dir, "tmp_alignment_test.dat")
    write_dat_file(NACA0012_W_U, NACA0012_W_L, dat_file, label="NACA 0012 CST")

    re = 100000.0  
    aoa = 0.0

    print("--- Running XFoil Alignment Test ---")
    print(f"Reynolds: {re:.1e} | AoA: {aoa} degrees")

    result = run_xfoil(dat_file, re, aoa)

    if result:
        cl, cd = result
        print("Success!")
        print(f"  Cl: {cl:.4f}")
        print(f"  Cd: {cd:.5f}")
        print(f"  L/D: {cl/cd:.2f}" if abs(cd) > 1e-6 else "  L/D: Inf")
    else:
        print("XFoil failed. Check the coordinates or XFoil installation.")