"""
Example: Optimize airfoil shape using CST parametrization.

Demonstrates gradient-based optimization of airfoil geometry
to maximize lift-to-drag ratio using differentiable CFD.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from turbodiff import FluidGrid
from turbodiff.core.airfoil import generate_cst_coords, thickness_at_x
from turbodiff.core.masking import create_airfoil_mask, interpolate_surface_to_grid
from turbodiff.core.loss_functions import (
    thickness_constraint_loss,
    crossover_validity_loss,
)
from turbodiff.core.optimization import create_optimizer


GRID_HEIGHT = 64
GRID_WIDTH = 128
CELL_SIZE = 0.1
NUM_SIM_STEPS = 80
NUM_ITERATIONS = 30
N_ORDER = 5
INFLOW_VELOCITY = 1.0
CHORD = 3.0


def create_airfoil_mask_from_weights(
    weights_upper, weights_lower, grid_x, grid_y, offset_x, offset_y, chord
):
    x_cst, y_upper_cst, y_lower_cst = generate_cst_coords(
        weights_upper, weights_lower, num_points=100
    )

    x_world = x_cst * chord + offset_x
    y_upper_world = y_upper_cst * chord + offset_y
    y_lower_world = y_lower_cst * chord + offset_y

    y_upper_grid = interpolate_surface_to_grid(grid_x, x_world, y_upper_world)
    y_lower_grid = interpolate_surface_to_grid(grid_x, x_world, y_lower_world)

    return create_airfoil_mask(
        grid_x,
        grid_y,
        y_upper_grid,
        y_lower_grid,
        x_min=offset_x,
        x_max=offset_x + chord,
        sharpness=50.0,
    )


def compute_loss_with_aux(params, grid_x, grid_y, offset_x, offset_y, chord):
    n_weights = N_ORDER + 1
    weights_upper = params[:n_weights]
    weights_lower = params[n_weights:]

    x_cst, y_upper, y_lower = generate_cst_coords(weights_upper, weights_lower)
    thickness = thickness_at_x(y_upper, y_lower)
    geo_loss = crossover_validity_loss(y_upper, y_lower) + thickness_constraint_loss(
        thickness, 0.06, 0.25
    )

    obstacle_mask = create_airfoil_mask_from_weights(
        weights_upper, weights_lower, grid_x, grid_y, offset_x, offset_y, chord
    )

    sim = FluidGrid(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        cell_size=CELL_SIZE,
        dt=0.05,
        diffusion=0.001,
        boundary_type=2,
        visualise=False,
    )

    state = sim.create_initial_state()
    state = sim.set_velocity_field(state, "wind tunnel")

    def step_fn(i, s):
        return sim.step(s)

    state = jax.lax.fori_loop(0, NUM_SIM_STEPS, step_fn, state)

    pressure = state.pressure.values
    cell_volume = CELL_SIZE**2

    # Compute spatial gradient of mask
    # mask=1 inside solid, so gradient points INWARD
    # Outward normal n = -grad(mask)
    grad_mask_x = jnp.zeros_like(obstacle_mask)
    grad_mask_y = jnp.zeros_like(obstacle_mask)
    grad_mask_x = grad_mask_x.at[:, 1:-1].set(
        (obstacle_mask[:, 2:] - obstacle_mask[:, :-2]) / (2 * CELL_SIZE)
    )
    grad_mask_y = grad_mask_y.at[1:-1, :].set(
        (obstacle_mask[2:, :] - obstacle_mask[:-2, :]) / (2 * CELL_SIZE)
    )

    # Pressure force on body: F = ∫ p * n_outward dA = -∫ p * grad(mask) dA
    # But we want force FROM fluid ON body, so negate: F = ∫ p * grad(mask) dA
    drag_force = jnp.sum(pressure * grad_mask_x) * cell_volume
    lift_force = jnp.sum(pressure * grad_mask_y) * cell_volume

    # Normalization for coefficients
    q = 0.5 * INFLOW_VELOCITY**2
    C_D = drag_force / (q * chord)
    C_L = lift_force / (q * chord)

    total_loss = jnp.abs(drag_force) + geo_loss

    return total_loss, (C_L, C_D, lift_force, drag_force)


def compute_loss(params, grid_x, grid_y, offset_x, offset_y, chord):
    loss, _ = compute_loss_with_aux(params, grid_x, grid_y, offset_x, offset_y, chord)
    return loss


def main():
    print("=" * 70)
    print("Airfoil Shape Optimization (CST Parametrization)")
    print("=" * 70)

    j_coords = jnp.arange(GRID_WIDTH, dtype=jnp.float32)
    i_coords = jnp.arange(GRID_HEIGHT, dtype=jnp.float32)
    j_grid, i_grid = jnp.meshgrid(j_coords, i_coords, indexing="xy")
    grid_x = (j_grid + 0.5) * CELL_SIZE
    grid_y = (i_grid + 0.5) * CELL_SIZE

    offset_x = 2.0
    offset_y = GRID_HEIGHT * CELL_SIZE / 2

    n_weights = N_ORDER + 1
    initial_upper = jnp.array([0.18, 0.22, 0.20, 0.18, 0.15, 0.12])
    initial_lower = jnp.array([-0.10, -0.08, -0.06, -0.05, -0.04, -0.03])
    initial_params = jnp.concatenate([initial_upper, initial_lower])

    print(f"\nGrid: {GRID_HEIGHT}x{GRID_WIDTH}, Sim steps: {NUM_SIM_STEPS}")
    print(f"CST order: {N_ORDER}, Weights per surface: {n_weights}")

    def loss_fn(p):
        return compute_loss(p, grid_x, grid_y, offset_x, offset_y, CHORD)

    def aux_fn(p):
        return compute_loss_with_aux(p, grid_x, grid_y, offset_x, offset_y, CHORD)

    grad_fn = jax.grad(loss_fn)

    opt_state, update_fn = create_optimizer("adam", learning_rate=0.005)
    params = initial_params

    print("\nStarting optimization...")
    print("-" * 70)

    loss_history = []
    cl_history = []
    cd_history = []

    for iteration in range(NUM_ITERATIONS):
        loss_value, (C_L, C_D, lift_force, drag_force) = aux_fn(params)
        gradients = grad_fn(params)
        loss_history.append(float(loss_value))
        cl_history.append(float(C_L))
        cd_history.append(float(C_D))

        cl_cd = float(C_L) / float(C_D) if abs(float(C_D)) > 1e-12 else 0.0
        print(
            f"Iter {iteration+1:2d} | CL/CD: {cl_cd:12.6f} | CL: {float(C_L):.6e} | "
            f"CD: {float(C_D):.6e} | Drag: {float(drag_force):.6e} | Lift: {float(lift_force):.6e}"
        )

        params, opt_state = update_fn(params, gradients, opt_state)

    print("-" * 70)
    print(f"\nDrag reduction: {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")

    final_upper = params[:n_weights]
    final_lower = params[n_weights:]

    # Visualization
    print("\nGenerating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x_init, y_upper_init, y_lower_init = generate_cst_coords(
        initial_upper, initial_lower
    )
    x_final, y_upper_final, y_lower_final = generate_cst_coords(
        final_upper, final_lower
    )

    ax.fill_between(
        x_init, y_lower_init, y_upper_init, alpha=0.3, color="red", label="Initial"
    )
    ax.plot(x_init, y_upper_init, "r--", linewidth=1.5)
    ax.plot(x_init, y_lower_init, "r--", linewidth=1.5)

    ax.fill_between(
        x_final,
        y_lower_final,
        y_upper_final,
        alpha=0.3,
        color="green",
        label="Optimized",
    )
    ax.plot(x_final, y_upper_final, "g-", linewidth=2)
    ax.plot(x_final, y_lower_final, "g-", linewidth=2)

    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Airfoil Shape Evolution")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = "airfoil_optimization.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
