"""
Quick test of geometry module - just visualize shapes
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import contourpy
from turbodiff.core.geometry import (
    sdf_circle,
    sdf_box,
    sdf_ellipse,
    sdf_parametric_blob,
)


def main():
    print("Testing geometry module...")
    print(f"contourpy version: {contourpy.__version__}")

    # Create a grid
    size = 64
    i_grid, j_grid = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")

    # Test different shapes
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Circle
    sdf = np.array(sdf_circle(j_grid, i_grid, center_x=32, center_y=32, radius=15))
    axes[0, 0].contour(sdf, levels=[0], colors="blue", linewidths=2, origin="lower")
    axes[0, 0].contourf(
        sdf, levels=[-1000, 0], colors=["lightblue"], alpha=0.5, origin="lower"
    )
    axes[0, 0].set_title("Circle")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].grid(True, alpha=0.3)

    # Box
    sdf = np.array(
        sdf_box(j_grid, i_grid, center_x=32, center_y=32, width=24, height=16)
    )
    axes[0, 1].contour(sdf, levels=[0], colors="red", linewidths=2, origin="lower")
    axes[0, 1].contourf(
        sdf, levels=[-1000, 0], colors=["lightcoral"], alpha=0.5, origin="lower"
    )
    axes[0, 1].set_title("Rectangle Box")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].grid(True, alpha=0.3)

    # Ellipse
    sdf = np.array(
        sdf_ellipse(j_grid, i_grid, center_x=32, center_y=32, a=18, b=10, angle=0.5)
    )
    axes[1, 0].contour(sdf, levels=[0], colors="green", linewidths=2, origin="lower")
    axes[1, 0].contourf(
        sdf, levels=[-1000, 0], colors=["lightgreen"], alpha=0.5, origin="lower"
    )
    axes[1, 0].set_title("Ellipse")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].grid(True, alpha=0.3)

    # Parametric blob
    params = jnp.array([0.2, 0.15, 0.1])
    sdf = np.array(
        sdf_parametric_blob(
            j_grid, i_grid, center_x=32, center_y=32, radius=12, params=params
        )
    )
    axes[1, 1].contour(sdf, levels=[0], colors="purple", linewidths=2, origin="lower")
    axes[1, 1].contourf(
        sdf, levels=[-1000, 0], colors=["plum"], alpha=0.5, origin="lower"
    )
    axes[1, 1].set_title("Parametric Blob")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    print("✓ All shapes rendered successfully!")
    plt.show()


if __name__ == "__main__":
    main()
