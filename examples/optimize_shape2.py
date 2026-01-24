"""
Example: Optimize a shape in a wind tunnel using PhiFlow-style differentiable masking.

Demonstrates the use of a Linear Regularized Heaviside function (linear ramp)
mask: clip(0.5 - sdf / cell_radius, 0, 1).
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from turbodiff import FluidGrid
from turbodiff.core.geometry import (
    sdf_circle,
    sdf_box,
    sdf_ellipse,
    sdf_parametric_blob,
)
from turbodiff.core.optimization import create_optimizer
from turbodiff.core.loss_functions import drag_loss


SHAPE_TYPE = "blob"


def get_shape_config(shape_type: str, grid_size: int):
    center = (grid_size // 3, grid_size // 2)

    if shape_type == "circle":
        initial_params = jnp.array([6.0])
        param_bounds = [(3.0, 10.0)]
        param_names = ["radius"]

    elif shape_type == "box":
        initial_params = jnp.array([16.0, 10.0])
        param_bounds = [(4.0, 20.0), (3.0, 15.0)]
        param_names = ["width", "height"]

    elif shape_type == "ellipse":
        initial_params = jnp.array([12.0, 4.0, 0.52])
        param_bounds = [(3.0, 15.0), (2.0, 10.0), (-1.57, 1.57)]
        param_names = ["semi_major", "semi_minor", "angle"]

    elif shape_type == "blob":
        initial_params = jnp.array([6.0, 0.2, 0.1, 0.05])
        param_bounds = [(3.0, 10.0), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]
        param_names = ["radius", "coef_1", "coef_2", "coef_3"]

    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    return initial_params, param_bounds, param_names, center


def create_sdf_from_params(shape_type: str, params, center):
    center_y, center_x = center

    if shape_type == "circle":
        radius = params[0]
        return lambda j, i: sdf_circle(j, i, center_x, center_y, radius)

    elif shape_type == "box":
        width, height = params[0], params[1]
        return lambda j, i: sdf_box(j, i, center_x, center_y, width, height)

    elif shape_type == "ellipse":
        a, b, angle = params[0], params[1], params[2]
        return lambda j, i: sdf_ellipse(j, i, center_x, center_y, a, b, angle)

    elif shape_type == "blob":
        radius = params[0]
        fourier_coefs = params[1:]
        return lambda j, i: sdf_parametric_blob(
            j, i, center_x, center_y, radius, fourier_coefs
        )

    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


def main():
    print("=" * 70)
    print(f"Shape Optimization (PhiFlow Mask): {SHAPE_TYPE.upper()}")
    print("=" * 70)

    grid_size = 64
    cell_size = 0.02
    dt = 0.01

    initial_params, param_bounds, param_names, center = get_shape_config(
        SHAPE_TYPE, grid_size
    )

    print(f"\nGrid: {grid_size}x{grid_size}, Cell size: {cell_size}")
    print(f"Shape type: {SHAPE_TYPE}")
    print(f"Initial parameters: {dict(zip(param_names, initial_params))}")

    def compute_loss(params):
        sdf_fn = create_sdf_from_params(SHAPE_TYPE, params, center)

        sim = FluidGrid(
            height=grid_size,
            width=grid_size,
            cell_size=cell_size,
            dt=dt,
            diffusion=0.0001,
            viscosity=0.0,
            boundary_type=1,
            sdf=sdf_fn,
            visualise=False,
        )

        state = sim.create_initial_state()
        state = sim.set_velocity_field(state, "wind tunnel")

        def step_fn(i, s):
            return sim.step(s)

        num_sim_steps = 100
        state = jax.lax.fori_loop(0, num_sim_steps, step_fn, state)

        j_coords = jnp.arange(grid_size, dtype=jnp.float32)
        i_coords = jnp.arange(grid_size, dtype=jnp.float32)
        j_grid, i_grid = jnp.meshgrid(j_coords, i_coords, indexing="xy")

        sdf_fn_grid = create_sdf_from_params(SHAPE_TYPE, params, center)
        sdf_values = sdf_fn_grid(j_grid, i_grid)

        # PhiFlow Style: Linear Ramp Mask
        # Inputs (j_grid, i_grid) are indices, so we are in grid index space (cell width = 1.0)
        # Bounding radius of 1x1 unit square is sqrt(2)/2 approx 0.707.
        cell_bounding_radius = jnp.sqrt(2.0) / 2.0

        # Formula: clip(0.5 - sdf / radius, 0, 1) to approximate volume fraction
        fraction_inside = 0.5 - sdf_values / cell_bounding_radius
        obstacle_mask = jnp.clip(fraction_inside, 0.0, 1.0)

        drag = drag_loss(state, obstacle_mask)

        # Approximate area calc for penalty
        if SHAPE_TYPE == "circle":
            area = jnp.pi * params[0] ** 2
        elif SHAPE_TYPE == "box":
            area = params[0] * params[1]
        elif SHAPE_TYPE == "ellipse":
            area = jnp.pi * params[0] * params[1]
        elif SHAPE_TYPE == "blob":
            area = jnp.pi * params[0] ** 2

        if SHAPE_TYPE == "circle":
            target_area = jnp.pi * initial_params[0] ** 2
        elif SHAPE_TYPE == "box":
            target_area = initial_params[0] * initial_params[1]
        elif SHAPE_TYPE == "ellipse":
            target_area = jnp.pi * initial_params[0] * initial_params[1]
        elif SHAPE_TYPE == "blob":
            target_area = jnp.pi * initial_params[0] ** 2

        area_penalty_weight = 1000.0
        area_penalty = area_penalty_weight * ((area - target_area) / target_area) ** 2

        return drag + area_penalty

    grad_fn = jax.grad(compute_loss)

    print("\nStarting optimization...")
    print("-" * 70)

    learning_rate = 0.02
    opt_state, update_fn = create_optimizer(
        "adam", learning_rate, beta1=0.9, beta2=0.999
    )

    params = initial_params
    num_iterations = 100
    loss_history = []

    for iteration in range(num_iterations):
        loss_value = compute_loss(params)
        gradients = grad_fn(params)
        loss_history.append(float(loss_value))

        if iteration % 1 == 0:
            grad_mag = float(jnp.linalg.norm(gradients))
            param_str = ", ".join(
                [f"{name}={val:.2f}" for name, val in zip(param_names, params)]
            )
            print(
                f"Iter {iteration:2d}: Drag={loss_value:8.4f}, {param_str}, |∇|={grad_mag:.6f}"
            )

        params, opt_state = update_fn(params, gradients, opt_state)

        params = jnp.array(
            [
                jnp.clip(params[i], bound[0], bound[1])
                for i, bound in enumerate(param_bounds)
            ]
        )

    print("-" * 70)
    print("\nOptimization complete!")
    print(f"Initial params: {dict(zip(param_names, initial_params))}")
    print(f"Final params:   {dict(zip(param_names, params))}")
    print(f"Drag reduction: {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")
    print(f"Initial drag: {loss_history[0]:.4f}, Final drag: {loss_history[-1]:.4f}")

    print("\nGenerating visualization...")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, "b-", linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Drag Force", fontsize=12)
    plt.title("Optimization Progress (PhiFlow Mask)", fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)

    i_grid, j_grid = jnp.meshgrid(
        jnp.arange(grid_size), jnp.arange(grid_size), indexing="ij"
    )

    sdf_fn_initial = create_sdf_from_params(SHAPE_TYPE, initial_params, center)
    sdf_initial = sdf_fn_initial(j_grid, i_grid)

    sdf_fn_final = create_sdf_from_params(SHAPE_TYPE, params, center)
    sdf_final = sdf_fn_final(j_grid, i_grid)

    plt.contour(
        j_grid,
        i_grid,
        sdf_initial,
        levels=[0],
        colors="red",
        linewidths=2,
        linestyles="--",
        label="Initial",
    )
    plt.contour(
        j_grid,
        i_grid,
        sdf_final,
        levels=[0],
        colors="green",
        linewidths=2,
        label="Optimized",
    )

    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title("Shape Evolution (PhiFlow Mask)", fontsize=14)
    plt.legend(fontsize=10)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = f"shape_optimization_phiflow_{SHAPE_TYPE}.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
