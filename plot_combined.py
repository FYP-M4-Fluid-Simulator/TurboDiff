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


def load_airfoil(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None
    with open(filepath, "r") as f:
        lines = f.readlines()
    coords = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 2:
            coords.append([float(parts[0]), float(parts[1])])
    coords = np.array(coords)
    return coords[:, 0], coords[:, 1]


def run_xfoil(airfoil_path, re, alphas, output_file):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            if len(f.readlines()) > 10:
                return True

    basename = os.path.basename(airfoil_path)
    output_basename = os.path.basename(output_file)
    tmp_airfoil = os.path.join(RESULTS_DIR, basename)
    if not os.path.exists(tmp_airfoil) or os.path.abspath(
        airfoil_path
    ) != os.path.abspath(tmp_airfoil):
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
        process.communicate(input=commands)
        return os.path.exists(output_file)
    except Exception as e:
        print(f"Error running XFoil: {e}")
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
        if start_reading and ("---" in line or not line.strip()):
            continue
        if start_reading:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    alpha, cl, cd = float(parts[0]), float(parts[1]), float(parts[2])
                    if cd > 0:
                        alphas.append(alpha)
                        cls.append(cl)
                        cds.append(cd)
                except ValueError:
                    continue
    return np.array(alphas), np.array(cls), np.array(cds)


# Configuration
re_list = [100000, 1000000, 6000000]
re_labels = ["10^5", "10^6", "6 \\times 10^6"]
aoas_opt = [0, 4, 8]
alpha_range = np.linspace(-2, 14, 33)
colors_opt = ["#FF5252", "#2196F3", "#4CAF50"]  # Red, Blue, Green


def plot_combined():
    for re, re_label in zip(re_list, re_labels):
        print(f"\n>>> Generating combined plots for Re = {re}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- LEFT: Geometry ---
        base_file = f"{TURBO_DIFF_DIR}/tmp_base_re_{re}.0_aoa_0.dat"
        xb, yb = load_airfoil(base_file)
        if xb is not None:
            ax1.plot(
                xb,
                yb,
                color="#333333",
                linestyle="--",
                label="Base Airfoil (NACA 0012)",
                alpha=0.5,
                linewidth=1.5,
            )

        for aoa, color in zip(aoas_opt, colors_opt):
            opt_file = f"{TURBO_DIFF_DIR}/best_airfoil_re_{re}.0_aoa_{aoa}.dat"
            xo, yo = load_airfoil(opt_file)
            if xo is not None:
                ax1.plot(
                    xo, yo, color=color, label=f"Optimized for $\\alpha = {aoa}^\\circ$"
                )

        ax1.set_title("(a) Geometry Comparison", pad=10)
        ax1.set_xlabel("Normalized Chord ($x/c$)")
        ax1.set_ylabel("Normalized Thickness ($y/c$)")
        ax1.set_aspect("equal")
        ax1.set_ylim(-0.3, 0.4)
        ax1.legend(loc="lower right", frameon=True, framealpha=0.9)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- RIGHT: Efficiency ---
        base_polar = os.path.join(RESULTS_DIR, f"base_re_{re}.txt")
        if run_xfoil(base_file, re, alpha_range, base_polar):
            a, cl, cd = parse_polar(base_polar)
            if a is not None and len(a) > 0:
                ax2.plot(
                    a,
                    cl / cd,
                    color="#333333",
                    linestyle="--",
                    label="Base Airfoil (NACA 0012)",
                    alpha=0.5,
                    linewidth=1.5,
                )

        for aoa, color in zip(aoas_opt, colors_opt):
            opt_file = f"{TURBO_DIFF_DIR}/best_airfoil_re_{re}.0_aoa_{aoa}.dat"
            opt_polar = os.path.join(RESULTS_DIR, f"opt_re_{re}_aoa_{aoa}.txt")
            if run_xfoil(opt_file, re, alpha_range, opt_polar):
                a, cl, cd = parse_polar(opt_polar)
                if a is not None and len(a) > 0:
                    ax2.plot(
                        a,
                        cl / cd,
                        color=color,
                        label=f"Optimized for $\\alpha = {aoa}^\\circ$",
                    )

        ax2.set_title("(b) Aerodynamic Efficiency", pad=10)
        ax2.set_xlabel(r"Angle of Attack $\alpha$ ($^\circ$)")
        ax2.set_ylabel("Lift-to-Drag Ratio ($C_L/C_D$)")
        ax2.set_xlim(-2, 14)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right", frameon=True, framealpha=0.9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.suptitle(
            f"Aerodynamic Optimization Results at $Re = {re_label}$",
            fontsize=16,
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        fn_re = f"{re:.0e}".replace("+0", "").replace("+", "")
        output_path = os.path.join(TURBO_DIFF_DIR, f"airfoil_re_{fn_re}.png")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Combined plot saved to {output_path}")
        plt.close()


if __name__ == "__main__":
    plot_combined()
