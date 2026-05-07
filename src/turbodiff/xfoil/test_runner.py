import os
import sys
import jax.numpy as jnp
from turbodiff.xfoil.runner import run_xfoil, write_dat_file

# Add the 'src' directory to sys.path so we can import 'turbodiff'
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# NACA 0012 Approximate CST Weights (from optimize_naca_suite.py)
NACA0012_W_U = jnp.array([0.1726, 0.1269, 0.1441, 0.1080, 0.1750, 0.1471])
NACA0012_W_L = -NACA0012_W_U


def test_naca_0012_alignment():
    # 1. Generate the DAT file exactly like the example script
    dat_file = os.path.join(current_dir, "tmp_alignment_test.dat")
    write_dat_file(NACA0012_W_U, NACA0012_W_L, dat_file, label="NACA 0012 CST")

    # 2. Use the exact parameters from the successful BASELINE run
    re = 1000000.0  # 1e6
    aoa = 0.0

    print("--- Running XFoil Alignment Test ---")
    print("Airfoil: Generated from CST Weights (NACA 0012)")
    print(f"Reynolds: {re:.1e}")
    print(f"AoA: {aoa} degrees")

    result = run_xfoil(dat_file, re, aoa)

    if result:
        cl, cd = result
        print("Success!")
        print(f"  Cl: {cl:.4f}")
        print(f"  Cd: {cd:.5f}")
        print(f"  L/D: {cl/cd:.2f}" if abs(cd) > 1e-6 else "  L/D: Inf")
    else:
        print("XFoil failed. Check the coordinates or XFoil installation.")


if __name__ == "__main__":
    test_naca_0012_alignment()
