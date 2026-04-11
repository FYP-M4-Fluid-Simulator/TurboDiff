import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Professional aesthetics for publication
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "figure.dpi": 300,
    }
)

XFOIL_PATH = "/Users/musab/Xfoil-for-Mac/bin/xfoil"
TURBO_DIFF_DIR = "/Users/musab/FYP/TurboDiff"
RESULTS_DIR = os.path.join(TURBO_DIFF_DIR, "xfoil_polars")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_xfoil(airfoil_path, re, alphas, output_file):
    if os.path.exists(output_file):
        # Check if file has data (not just header)
        with open(output_file, "r") as f:
            if len(f.readlines()) > 10:
                return True

    basename = os.path.basename(airfoil_path)
    output_basename = os.path.basename(output_file)
    tmp_airfoil = os.path.join(RESULTS_DIR, basename)
    if not os.path.exists(tmp_airfoil) or airfoil_path != tmp_airfoil:
        import shutil

        shutil.copy(airfoil_path, tmp_airfoil)

    commands = f"""LOAD {basename}
PANE
OPER
ITER 200
VISC {re}
PACC
{output_basename}

"""
    for a in alphas:
        commands += f"ALFA {a}\n"

    commands += "QUIT\n"

    try:
        current_env = os.environ.copy()
        current_env["DISPLAY"] = ":0"
        process = subprocess.Popen(
            XFOIL_PATH,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=current_env,
            cwd=RESULTS_DIR,
        )
        out, err = process.communicate(input=commands)

        success = os.path.exists(output_file)
        return success
    except Exception as e:
        print(f"Error running XFoil for {basename}: {e}")
        return False


def parse_polar(polar_file):
    alphas, cls, cds = [], [], []
    if not os.path.exists(polar_file):
        return None

    with open(polar_file, "r") as f:
        lines = f.readlines()

    start_reading = False
    for line in lines:
        if "alpha" in line and "CL" in line and "CD" in line:
            start_reading = True
            continue
        if start_reading and "---" in line:
            continue
        if start_reading:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    alpha = float(parts[0])
                    cl = float(parts[1])
                    cd = float(parts[2])
                    if cd > 0:
                        alphas.append(alpha)
                        cls.append(cl)
                        cds.append(cd)
                except ValueError:
                    continue

    return np.array(alphas), np.array(cls), np.array(cds)


# Configuration
re_list = [100000, 1000000, 6000000]
aoas_opt = [0, 4, 8]
alpha_range = np.linspace(-2, 14, 33)

# Premium Color Palette
colors_opt = ["#FF5252", "#2196F3", "#4CAF50"]  # Vibrant Red, Blue, Green

for re in re_list:
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Base Airfoil
    re_label = f"10^{int(np.log10(re))}" if re != 6000000 else "6 \\times 10^6"

    base_file = f"/Users/musab/FYP/TurboDiff/tmp_base_re_{re}.0_aoa_0.dat"
    base_polar = os.path.join(RESULTS_DIR, f"base_re_{re}.txt")
    if run_xfoil(base_file, re, alpha_range, base_polar):
        a, cl, cd = parse_polar(base_polar)
        if a is not None and len(a) > 0:
            ax.plot(
                a,
                cl / cd,
                color="#333333",
                linestyle="--",
                label="Base Airfoil (NACA 0012)",
                alpha=0.5,
                linewidth=1.5,
            )

    # 2. Optimized Airfoils
    for target_aoa, color in zip(aoas_opt, colors_opt):
        opt_file = (
            f"/Users/musab/FYP/TurboDiff/best_airfoil_re_{re}.0_aoa_{target_aoa}.dat"
        )
        opt_polar = os.path.join(RESULTS_DIR, f"opt_re_{re}_aoa_{target_aoa}.txt")

        print(f"Running/Checking XFoil for Re={re} Optimized for AoA {target_aoa}...")
        if run_xfoil(opt_file, re, alpha_range, opt_polar):
            a, cl, cd = parse_polar(opt_polar)
            if a is not None and len(a) > 0:
                ax.plot(
                    a,
                    cl / cd,
                    color=color,
                    label=f"Optimized for $\\alpha = {target_aoa}^\\circ$",
                )

    ax.set_title(f"Aerodynamic Efficiency ($C_L/C_D$) at $Re = {re_label}$", pad=15)
    ax.set_xlabel(r"Angle of Attack $\alpha$ ($^\circ$)")
    ax.set_ylabel("Lift-to-Drag Ratio ($C_L/C_D$)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.set_xlim(-2, 14)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    # Mapping for consistent filename labels
    fn_re = "1e+05" if re == 100000 else ("1e+06" if re == 1000000 else "6e+06")
    output_plot = os.path.join(TURBO_DIFF_DIR, f"efficiency_re_{fn_re}_all_models.png")
    plt.savefig(output_plot, bbox_inches="tight")
    print(f"Performance plot saved to {output_plot}")
    plt.close()
