"""
plot_improvement_bars.py

Grouped bar charts showing % lift-to-drag improvement of the optimised
airfoils over the baseline at every (Re, AoA) combination.

  x-axis groups : Reynolds number  (Micro / Community / Utility scale)
  bars per group: angle of attack used during optimisation
                  NACA 0012 → α = 4°, 8°
                  S809      → α = 0°, 4°

Usage:
    python plot_improvement_bars.py          # NACA 0012 (default)
    python plot_improvement_bars.py s809     # S809
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Poster palette (mirrors plot_combined_poster.py) ──────────────────────────
POSTER_DARK = "#194E76"  # Deep blue
POSTER_BG = "#196B89"  # Main background
POSTER_CYAN = "#6EC6D9"  # Bright cyan
POSTER_WHITE = "#FFFFFF"  # White text
POSTER_LIGHT = "#EEF5FC"  # Light blue
POSTER_GRID = "#CCE2F5"  # Grid lines

# ── Colour modes ─────────────────────────────────────────────────────────────
# "accent"  → colourful bars matching the geometry line colours
# "blue"    → poster blue tones for a fully cohesive look
#
# Set via second CLI argument:  python plot_improvement_bars.py naca blue
#                               python plot_improvement_bars.py naca accent  (default)

BAR_COLORS_ACCENT = {
    "naca": ["#ADFF2F", "#FFD700"],  # lime, gold
    "s809": ["#ADFF2F", "#FFD700"],
}

BAR_COLORS_BLUE = {
    "naca": [POSTER_CYAN, POSTER_DARK],  # cyan, dark blue
    "s809": [POSTER_CYAN, POSTER_DARK],
}

HATCH_PATTERNS = ["", "///"]  # solid / hatched for visual distinction

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
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
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
    }
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# ── Configuration ─────────────────────────────────────────────────────────────
SUITE = sys.argv[1].lower() if len(sys.argv) > 1 else "naca"
COLOR_MODE = sys.argv[2].lower() if len(sys.argv) > 2 else "accent"  # "accent" | "blue"
BAR_COLORS = BAR_COLORS_BLUE if COLOR_MODE == "blue" else BAR_COLORS_ACCENT

# ── Paths ─────────────────────────────────────────────────────────────────────
TURBO_DIFF_DIR = "/Users/musab/FYP/TurboDiff"
RESULTS_DIR = os.path.join(
    TURBO_DIFF_DIR, "naca_results" if SUITE == "naca" else "s809_results"
)

RE_LIST = [100_000, 1_000_000, 6_000_000]
RE_LABELS = [
    "Micro\n" + r"$Re=10^5$",
    "Community\n" + r"$Re=10^6$",
    "Utility\n" + r"$Re=6\!\times\!10^6$",
]

# AoAs to display (those actually optimised)
SUITE_CONFIG = {
    "naca": {
        "aoas": [4, 8],
        "base_prefix": "tmp_base",
        "opt_prefix": "best_airfoil",
        "base_label": "NACA 0012",
    },
    "s809": {
        "aoas": [0, 4],
        "base_prefix": "tmp_s809_base",
        "opt_prefix": "best_s809",
        "base_label": "S809",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def parse_polar(polar_file):
    """Return (alphas, cls, cds) arrays from an XFoil polar file."""
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


def parse_polar_at_aoa(polar_file, target_aoa):
    """Return (cl, cd) at the closest alpha to target_aoa (within ±0.1°).
    Matches the logic used in create_optimization_summary.py."""
    if not os.path.exists(polar_file):
        return None
    best_match = None
    min_diff = float("inf")
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
                    alpha, cl, cd = float(parts[0]), float(parts[1]), float(parts[2])
                    diff = abs(alpha - target_aoa)
                    if diff < 0.1 and diff < min_diff:
                        min_diff = diff
                        best_match = (cl, cd)
                except ValueError:
                    continue
    return best_match


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────


def collect_improvements(suite, cfg):
    """
    Returns a 2-D list  improvements[re_idx][aoa_idx]  of % L/D improvement.
    Uses L/D at the exact target AoA — same method as create_optimization_summary.py.
    None means data unavailable / baseline near-zero.
    """
    aoas = cfg["aoas"]
    data = []

    for re in RE_LIST:
        row = []
        for aoa in aoas:
            base_polar = os.path.join(
                RESULTS_DIR, f"{cfg['base_prefix']}_re_{float(re)}_aoa_{aoa}.txt"
            )
            opt_polar = os.path.join(
                RESULTS_DIR, f"{cfg['opt_prefix']}_re_{float(re)}_aoa_{aoa}.txt"
            )
            base_perf = parse_polar_at_aoa(base_polar, aoa)
            opt_perf = parse_polar_at_aoa(opt_polar, aoa)

            if base_perf and opt_perf:
                base_ld = base_perf[0] / max(base_perf[1], 1e-6)
                opt_ld = opt_perf[0] / max(opt_perf[1], 1e-6)
                if abs(base_ld) < 0.1:  # near-zero baseline → report as 0%
                    imp = 0.0
                else:
                    imp = (opt_ld - base_ld) / base_ld * 100.0
            else:
                imp = None

            row.append(imp)
        data.append(row)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────


def plot_bars(suite, cfg, improvements):
    aoas = cfg["aoas"]
    colors = BAR_COLORS[suite]
    n_re = len(RE_LIST)
    n_aoa = len(aoas)

    bar_w = 0.28  # width of one bar
    spacing = 0.12  # gap between bar groups
    group_w = n_aoa * bar_w + spacing
    x_centers = np.arange(n_re) * group_w

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="none")
    ax.set_facecolor("none")

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(POSTER_CYAN)
        ax.spines[spine].set_linewidth(1.5)

    # Draw zero reference line
    ax.axhline(0, color=POSTER_CYAN, linewidth=1.5, linestyle="-", alpha=0.5)

    bars_drawn = []  # collect for legend

    for aoa_i, (aoa, color, hatch) in enumerate(zip(aoas, colors, HATCH_PATTERNS)):
        offsets = x_centers + (aoa_i - (n_aoa - 1) / 2) * bar_w
        vals = [improvements[re_i][aoa_i] for re_i in range(n_re)]

        for re_i, (x, v) in enumerate(zip(offsets, vals)):
            if v is None:
                # Draw a striped "N/A" bar
                ax.bar(
                    x,
                    5,
                    width=bar_w,
                    color="none",
                    edgecolor=POSTER_CYAN,
                    linewidth=0.8,
                    hatch="xxx",
                    alpha=0.4,
                )
                ax.text(
                    x,
                    6,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color=POSTER_WHITE,
                    alpha=0.6,
                )
            else:
                # Value label on each bar
                label_y = v + (1.5 if v >= 0 else -3.5)
                va = "bottom" if v >= 0 else "top"
                ax.text(
                    x,
                    label_y,
                    f"{v:+.1f}%",
                    ha="center",
                    va=va,
                    fontsize=18,
                    fontweight="bold",
                    color=POSTER_WHITE,
                    zorder=4,
                )

        # Invisible proxy bar for legend
        proxy = plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color,
            edgecolor=POSTER_WHITE,
            linewidth=1.0,
            hatch=hatch,
            alpha=0.9,
        )
        bars_drawn.append((proxy, rf"$\alpha = {aoa}^\circ$"))

    # ── Axes decoration ───────────────────────────────────────────────────────
    ax.set_xticks(x_centers)
    ax.set_xticklabels(RE_LABELS, fontsize=16, linespacing=1.4)
    ax.set_ylabel(
        "Peak $C_L/C_D$ Improvement  (%)", fontsize=20, labelpad=15, color=POSTER_WHITE
    )
    ax.set_xlim(x_centers[0] - group_w * 0.55, x_centers[-1] + group_w * 0.55)

    ax.yaxis.grid(
        True, linestyle="--", linewidth=0.7, color=POSTER_GRID, alpha=0.9, zorder=0
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=6, width=1.2, color=POSTER_CYAN)

    # ── Title ─────────────────────────────────────────────────────────────────
    suite_name = "NACA 0012" if suite == "naca" else "S809"
    ax.set_title(
        f"{suite_name}  —  $C_L/C_D$ Improvement by Wind Scale & AoA",
        fontsize=24,
        fontweight="bold",
        color=POSTER_CYAN,
        pad=45,
        loc="center",
    )

    # ── Legend — placed below the axes so it never overlaps bars ──────────────
    handles, labels = zip(*bars_drawn)
    leg = ax.legend(
        handles,
        labels,
        title="Optimisation AoA",
        title_fontsize=18,
        frameon=True,
        fontsize=18,
        edgecolor=POSTER_CYAN,
        facecolor=POSTER_BG,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),  # below the axes
        ncol=len(aoas),  # all entries in one row
    )
    leg.get_frame().set_linewidth(1.2)
    leg.get_title().set_color(POSTER_CYAN)
    leg.get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)  # reserve space for below-axes legend

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(TURBO_DIFF_DIR, f"{suite}_improvement_bars.png")
    plt.savefig(out_path, bbox_inches="tight", transparent=True, dpi=300)
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    cfg = SUITE_CONFIG[SUITE]
    improvements = collect_improvements(SUITE, cfg)

    print(f"\n>>> {SUITE.upper()} improvement data:")
    for re, re_lbl, row in zip(RE_LIST, RE_LABELS, improvements):
        print(f"  Re={re:>9,}  →  {row}")

    plot_bars(SUITE, cfg, improvements)


if __name__ == "__main__":
    main()
