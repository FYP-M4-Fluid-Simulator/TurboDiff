import matplotlib.pyplot as plt
import numpy as np
import os

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


def load_airfoil(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Skip the first line (header)
    coords = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 2:
            coords.append([float(parts[0]), float(parts[1])])

    coords = np.array(coords)
    return coords[:, 0], coords[:, 1]


# Configuration
res = [100000.0, 1000000.0, 6000000.0]
labels = ["10^5", "10^6", "6 \\times 10^6"]
aoas_opt = [0, 4, 8]

# Premium Color Palette
colors_opt = ["#FF5252", "#2196F3", "#4CAF50"]  # Vibrant Red, Blue, Green

for re, label_re in zip(res, labels):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Base Airfoil (NACA 0012)
    base_file = f"/Users/musab/FYP/TurboDiff/tmp_base_re_{re}_aoa_0.dat"
    x_base, y_base = load_airfoil(base_file)
    if x_base is not None:
        ax.plot(
            x_base,
            y_base,
            color="#333333",
            linestyle="--",
            label="Base Airfoil (NACA 0012)",
            alpha=0.5,
            linewidth=1.5,
        )

    # Plot Optimized Airfoils for each target AoA
    for aoa, color in zip(aoas_opt, colors_opt):
        filepath = f"/Users/musab/FYP/TurboDiff/best_airfoil_re_{re}_aoa_{aoa}.dat"
        x, y = load_airfoil(filepath)
        if x is not None:
            ax.plot(x, y, color=color, label=f"Optimized for $\\alpha = {aoa}^\\circ$")

    ax.set_title(f"Airfoil Geometry Evolution at $Re = {label_re}$", pad=15)
    ax.set_xlabel("Normalized Chord ($x/c$)")
    ax.set_ylabel("Normalized Thickness ($y/c$)")

    # Make diagram taller and provide space for legend
    ax.set_aspect("equal")
    ax.set_ylim(-0.3, 0.4)

    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Mapping for consistent filename labels
    fn_re = "1e5" if re == 100000.0 else ("1e6" if re == 1000000.0 else "6e6")
    output_path = f"/Users/musab/FYP/TurboDiff/geometry_re_{fn_re}_all.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Geometry plot saved to {output_path}")
    plt.close()
