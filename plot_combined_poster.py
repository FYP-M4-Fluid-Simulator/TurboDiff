"""
plot_combined_poster.py

Recreates the aerodynamic optimisation plots with:
  - Vertical layout  (geometry top, efficiency bottom) per Reynolds number
  - Poster-aligned visual theme (TurboDiff blue/white palette, Roboto font)
  - Wind-condition headings: Micro / Community / Utility scale
  - Larger, readable axis labels and titles

Usage:
    python plot_combined_poster.py          # NACA 0012 (default)
    python plot_combined_poster.py s809     # S809
"""

import os
import subprocess
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

# ── Poster palette ────────────────────────────────────────────────────────────
POSTER_DARK = "#194E76"  # Deep blue for headers/accents
POSTER_BG = "#196B89"  # Main background color
POSTER_CYAN = "#6EC6D9"  # Bright cyan for primary highlights
POSTER_WHITE = "#FFFFFF"  # White for text
POSTER_LIGHT = "#EEF5FC"  # Secondary light blue
POSTER_GRID = "#CCE2F5"  # Grid lines (semi-transparent)

# Line colours for the optimised airfoils (brighter for dark background)
OPT_COLORS = [POSTER_CYAN, "#ADFF2F", "#FFD700"]  # cyan · lime · gold

BASE_COLOR = "#EEF5FC"  # Light blue for baseline

# ── Typography ────────────────────────────────────────────────────────────────
rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Inter",
            "DejaVu Sans",
            "Liberation Sans",
            "Arial",
            "Helvetica",
        ],
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth": 4.0,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": POSTER_BG,
        "figure.facecolor": POSTER_BG,
        "axes.edgecolor": POSTER_CYAN,
        "axes.labelcolor": POSTER_WHITE,
        "xtick.color": POSTER_WHITE,
        "ytick.color": POSTER_WHITE,
        "text.color": POSTER_WHITE,
        "grid.color": POSTER_GRID,
        "grid.linewidth": 0.8,
        "legend.framealpha": 0.15,
        "legend.edgecolor": POSTER_WHITE,
    }
)

# ── Configuration ─────────────────────────────────────────────────────────────
SUITE = sys.argv[1].lower() if len(sys.argv) > 1 else "naca"

# ── Paths ─────────────────────────────────────────────────────────────────────
XFOIL_PATH = "/Users/musab/Xfoil-for-Mac/bin/xfoil"
TURBO_DIFF_DIR = "/Users/musab/FYP/TurboDiff"
RESULTS_DIR = os.path.join(
    TURBO_DIFF_DIR, "naca_results" if SUITE == "naca" else "s809_results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Wind-condition labelling ──────────────────────────────────────────────────
# Re → (scale label, descriptive subtitle)
WIND_CONDITIONS = {
    100_000: ("Micro-Scale Wind", r"$Re = 10^5$  |  Small turbines, UAVs"),
    1_000_000: ("Community-Scale Wind", r"$Re = 10^6$  |  Mid-size turbines"),
    6_000_000: (
        "Utility-Scale Wind",
        r"$Re = 6\times10^6$  |  Large commercial turbines",
    ),
}

re_list = [100_000, 1_000_000, 6_000_000]
re_labels = [r"10^5", r"10^6", r"6 \times 10^6"]
aoas_opt = [0, 4] if SUITE == "s809" else [0, 4, 8]
alpha_range = np.linspace(-2, 14, 33)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_airfoil(filepath):
    if not os.path.exists(filepath):
        return None, None
    with open(filepath) as f:
        lines = f.readlines()
    coords = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 2:
            try:
                coords.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    if not coords:
        return None, None
    coords = np.array(coords)
    return coords[:, 0], coords[:, 1]


def run_xfoil(airfoil_path, re, alphas, output_file):
    if os.path.exists(output_file):
        with open(output_file) as f:
            # Check for at least 20 lines to ensure it's a full curve, not just a single-point polar
            if len(f.readlines()) > 20:
                return True

    basename = os.path.basename(airfoil_path)
    output_basename = os.path.basename(output_file)
    tmp_airfoil = os.path.join(RESULTS_DIR, basename)
    if not os.path.exists(tmp_airfoil) or os.path.abspath(
        airfoil_path
    ) != os.path.abspath(tmp_airfoil):
        import shutil

        shutil.copy(airfoil_path, tmp_airfoil)

    commands = (
        f"LOAD {basename}\nPANE\nOPER\nITER 200\nVISC {re}\nPACC\n{output_basename}\n\n"
    )
    for a in alphas:
        commands += f"ALFA {a}\n"
    commands += "QUIT\n"

    try:
        env = os.environ.copy()
        env["DISPLAY"] = ":0"
        proc = subprocess.Popen(
            XFOIL_PATH,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=RESULTS_DIR,
        )
        proc.communicate(input=commands)
        return os.path.exists(output_file)
    except Exception as e:
        print(f"  [XFoil error] {e}")
        return False


def parse_polar(polar_file):
    alphas, cls, cds = [], [], []
    if not os.path.exists(polar_file):
        return None, None, None
    with open(polar_file) as f:
        lines = f.readlines()
    reading = False
    for line in lines:
        if "alpha" in line and "CL" in line and "CD" in line:
            reading = True
            continue
        if reading and ("---" in line or not line.strip()):
            continue
        if reading:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    a, cl, cd = float(parts[0]), float(parts[1]), float(parts[2])
                    if cd > 0:
                        alphas.append(a)
                        cls.append(cl)
                        cds.append(cd)
                except ValueError:
                    continue
    if not alphas:
        return None, None, None
    return np.array(alphas), np.array(cls), np.array(cds)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────


def style_ax(ax):
    """Apply consistent poster styling to an Axes."""
    ax.set_facecolor(POSTER_BG)
    ax.grid(True, linestyle="--", linewidth=0.7, color=POSTER_GRID, alpha=0.3)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, color=POSTER_CYAN)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(POSTER_CYAN)
        ax.spines[spine].set_linewidth(1.2)


def draw_section_banner(fig, ax, title, subtitle, color=POSTER_DARK):
    """Draw a coloured banner above an axes as the section heading."""
    # We annotate above the axes in figure coordinates
    bbox = ax.get_position()
    x0 = bbox.x0
    width = bbox.width
    y_top = bbox.y1

    # Banner rectangle
    banner_h = 0.045
    rect = FancyBboxPatch(
        (x0, y_top + 0.005),
        width,
        banner_h,
        boxstyle="round,pad=0.005",
        transform=fig.transFigure,
        facecolor=color,
        edgecolor=POSTER_CYAN,
        linewidth=1.2,
        zorder=5,
        clip_on=False,
    )
    fig.add_artist(rect)

    # Title text
    fig.text(
        x0 + width / 2,
        y_top + 0.005 + banner_h / 2 + 0.007,
        title,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=POSTER_WHITE,
        zorder=6,
        transform=fig.transFigure,
    )
    # Subtitle text
    fig.text(
        x0 + width / 2,
        y_top + 0.005 + banner_h / 2 - 0.010,
        subtitle,
        ha="center",
        va="center",
        fontsize=16,
        color=POSTER_CYAN,
        zorder=6,
        transform=fig.transFigure,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main plot function
# ─────────────────────────────────────────────────────────────────────────────


def plot_vertical(
    suite,
    re,
    re_label,
    wind_title,
    wind_subtitle,
    base_prefix,
    opt_prefix,
    base_label,
    aoas,
    colors,
):

    print(f"\n>>> Generating poster plot for {suite.upper()} — {wind_title}  Re={re}")

    # Tall figure: geometry (top) + efficiency (bottom)
    fig = plt.figure(figsize=(8, 14), facecolor=POSTER_BG)
    gs = GridSpec(
        2, 1, figure=fig, hspace=0.35, top=0.94, bottom=0.22, left=0.14, right=0.95
    )

    ax_geom = fig.add_subplot(gs[0])
    ax_eff = fig.add_subplot(gs[1])

    # Efficiency panel gets full styling; geometry panel is decoration-free
    style_ax(ax_eff)

    # ── PANEL A — Geometry ───────────────────────────────────────────────────
    base_file = os.path.join(RESULTS_DIR, f"{base_prefix}_re_{float(re)}_aoa_0.dat")
    xb, yb = load_airfoil(base_file)
    if xb is not None:
        ax_geom.plot(
            xb,
            yb,
            color=BASE_COLOR,
            linestyle="--",
            linewidth=1.8,
            label=base_label,
            alpha=0.75,
        )

    for aoa, color in zip(aoas, colors):
        opt_file = os.path.join(
            RESULTS_DIR, f"{opt_prefix}_re_{float(re)}_aoa_{aoa}.dat"
        )
        xo, yo = load_airfoil(opt_file)
        if xo is not None:
            ax_geom.plot(
                xo,
                yo,
                color=color,
                linewidth=2.2,
                zorder=10,
                label=rf"Optimised  $\alpha = {aoa}^\circ$",
            )

    # Strip all axes decoration — just the airfoil shapes
    ax_geom.set_title(
        "(a)  Geometry Comparison",
        fontsize=22,
        fontweight="bold",
        color=POSTER_CYAN,
        pad=10,
        loc="left",
    )
    ax_geom.set_aspect("equal")
    ax_geom.set_xlim(-0.05, 1.05)
    ax_geom.set_ylim(-0.30, 0.42)
    ax_geom.set_facecolor(POSTER_BG)
    ax_geom.axis("off")  # hide all spines, ticks, and labels
    # No per-axes legend — shared legend added at figure level below

    # ── PANEL B — Aerodynamic Efficiency ────────────────────────────────────
    base_polar = os.path.join(RESULTS_DIR, f"base_full_re_{float(re)}.txt")
    if run_xfoil(base_file, re, alpha_range, base_polar):
        a, cl, cd = parse_polar(base_polar)
        if a is not None and len(a) > 0:
            ax_eff.plot(
                a,
                cl / cd,
                color=BASE_COLOR,
                linestyle="--",
                linewidth=3.0,
                label=base_label,
                alpha=0.75,
            )

    for aoa, color in zip(aoas, colors):
        opt_file = os.path.join(
            RESULTS_DIR, f"{opt_prefix}_re_{float(re)}_aoa_{aoa}.dat"
        )
        opt_polar = os.path.join(
            RESULTS_DIR, f"opt_{suite}_re_{float(re)}_aoa_{aoa}.txt"
        )
        if run_xfoil(opt_file, re, alpha_range, opt_polar):
            a, cl, cd = parse_polar(opt_polar)
            if a is not None and len(a) > 0:
                ax_eff.plot(
                    a,
                    cl / cd,
                    color=color,
                    linewidth=4.0,
                    label=rf"Optimised  $\alpha = {aoa}^\circ$",
                )

    ax_eff.set_title(
        "(b)  Aerodynamic Efficiency",
        fontsize=22,
        fontweight="bold",
        color=POSTER_CYAN,
        pad=10,
        loc="left",
    )
    ax_eff.set_xlabel(r"Angle of Attack   $\alpha$  (°)", labelpad=6)
    ax_eff.set_ylabel(r"Lift-to-Drag Ratio   $C_L / C_D$", labelpad=8)
    ax_eff.set_xlim(-2, 14)
    ax_eff.xaxis.set_major_locator(ticker.MultipleLocator(2))

    # ── Shared figure-level legend (applies to both panels) ─────────────────
    # Collect handles from the geometry axes (labels are identical in both)
    handles, labels = ax_geom.get_legend_handles_labels()
    ncols = min(len(handles), 2)  # 2 columns keeps it compact
    fig_legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.545, 0.02),
        ncol=ncols,
        labelspacing=1.5,
        handletextpad=1.0,
        columnspacing=2.0,
        frameon=True,
        fontsize=18,
        edgecolor=POSTER_CYAN,
        facecolor=POSTER_BG,
        title="Airfoil configuration",
        title_fontsize=18,
    )
    fig_legend.get_frame().set_linewidth(0.9)
    fig_legend.get_title().set_color(POSTER_CYAN)
    fig_legend.get_title().set_fontweight("bold")

    # ── Save ─────────────────────────────────────────────────────────────────
    fn_re = f"{re:.0e}".replace("+0", "").replace("+", "")
    out_path = os.path.join(TURBO_DIFF_DIR, f"{suite}_poster_re_{fn_re}.png")
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    if SUITE == "s809":
        base_prefix = "tmp_s809_base"
        opt_prefix = "best_s809"
        base_label = "Baseline  (S809)"
    else:
        base_prefix = "tmp_base"
        opt_prefix = "best_airfoil"
        base_label = "Baseline  (NACA 0012)"

    colors = OPT_COLORS[: len(aoas_opt)]

    for re, re_label in zip(re_list, re_labels):
        wind_title, wind_subtitle = WIND_CONDITIONS[re]
        plot_vertical(
            suite=SUITE,
            re=re,
            re_label=re_label,
            wind_title=wind_title,
            wind_subtitle=wind_subtitle,
            base_prefix=base_prefix,
            opt_prefix=opt_prefix,
            base_label=base_label,
            aoas=aoas_opt,
            colors=colors,
        )


if __name__ == "__main__":
    main()
