"""
Example: Optimize airfoil shape using CST parametrization.

Demonstrates gradient-based optimization of airfoil geometry
to maximize lift-to-drag ratio using differentiable CFD.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from turbodiff import FluidGrid
from turbodiff.core.airfoil import generate_cst_coords, thickness_at_x, cst_to_closed_contour
from turbodiff.core.masking import soft_sigmoid_mask
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


def compute_airfoil_sdf(
    grid_x, grid_y, weights_upper, weights_lower, offset_x, offset_y, chord
):
    """
    Compute signed distance field for CST airfoil.
    
    Returns negative values inside the airfoil, positive outside.
    Uses the contour points to compute approximate distances.
    """
    # Generate CST coordinates
    x_cst, y_upper_cst, y_lower_cst = generate_cst_coords(
        weights_upper, weights_lower, num_points=200
    )
    
    # Create closed contour
    x_contour, y_contour = cst_to_closed_contour(x_cst, y_upper_cst, y_lower_cst)
    
    # Transform to world coordinates
    x_world = x_contour * chord + offset_x
    y_world = y_contour * chord + offset_y
    
    # Stack contour points: shape (num_points, 2)
    contour_points = jnp.stack([x_world, y_world], axis=1)
    
    # Flatten grid points
    grid_points = jnp.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    
    # Compute distances to all contour points for each grid point
    # grid_points: (H*W, 2), contour_points: (num_contour, 2)
    # Broadcast to (H*W, num_contour, 2)
    diff = grid_points[:, None, :] - contour_points[None, :, :]
    distances = jnp.sqrt(jnp.sum(diff**2, axis=2))  # (H*W, num_contour)
    
    # Minimum distance to surface for each grid point
    min_dist = jnp.min(distances, axis=1)  # (H*W,)
    
    # Determine inside/outside using point-in-polygon test
    # Simple approach: check if point is between upper and lower surfaces
    # For each x position, interpolate upper and lower y values
    grid_x_norm = (grid_x - offset_x) / chord
    
    # Interpolate upper and lower surfaces at grid x positions
    y_upper_interp = jnp.interp(grid_x_norm.flatten(), x_cst, y_upper_cst)
    y_lower_interp = jnp.interp(grid_x_norm.flatten(), x_cst, y_lower_cst)
    
    # Transform to world coordinates
    y_upper_world = y_upper_interp * chord + offset_y
    y_lower_world = y_lower_interp * chord + offset_y
    
    # Check if point is inside: between surfaces and within chord
    grid_y_flat = grid_y.flatten()
    grid_x_flat = grid_x.flatten()
    
    is_between_surfaces = (grid_y_flat >= y_lower_world) & (grid_y_flat <= y_upper_world)
    is_within_chord = (grid_x_flat >= offset_x) & (grid_x_flat <= offset_x + chord)
    is_inside = is_between_surfaces & is_within_chord
    
    # Apply sign: negative inside, positive outside
    sdf = jnp.where(is_inside, -min_dist, min_dist)
    
    return sdf.reshape(grid_x.shape)


def create_airfoil_mask_from_sdf(
    weights_upper, weights_lower, grid_x, grid_y, offset_x, offset_y, chord, sharpness=50.0
):
    """Create soft mask from SDF using sigmoid."""
    sdf = compute_airfoil_sdf(
        grid_x, grid_y, weights_upper, weights_lower, offset_x, offset_y, chord
    )
    # Convert SDF to mask: negative (inside) -> 1, positive (outside) -> 0
    return soft_sigmoid_mask(sdf, sharpness)


def compute_loss_with_aux(params, grid_x, grid_y, offset_x, offset_y, chord, sim):
    n_weights = N_ORDER + 1
    weights_upper = params[:n_weights]
    weights_lower = params[n_weights:]

    x_cst, y_upper, y_lower = generate_cst_coords(weights_upper, weights_lower)
    thickness = thickness_at_x(y_upper, y_lower)
    geo_loss = crossover_validity_loss(y_upper, y_lower) + thickness_constraint_loss(
        thickness, 0.06, 0.25
    )

    # Compute SDF and convert to mask
    sdf = compute_airfoil_sdf(
        grid_x, grid_y, weights_upper, weights_lower, offset_x, offset_y, chord
    )
    obstacle_mask = soft_sigmoid_mask(sdf, sharpness=1000.0)

    # Create initial state and update solid_mask
    state = sim.create_initial_state()
    # Update the solid_mask in the state to use our computed mask
    state = state.__class__(
        density=state.density,
        velocity=state.velocity,
        pressure=state.pressure,
        solid_mask=obstacle_mask,
        sources=state.sources,
        time=state.time,
        step=state.step,
    )
    state = sim.set_velocity_field(state, "wind tunnel")

    # Run simulation using scan for efficiency and differentiation support
    def simulation_step(carry_state, _):
        """Single simulation step - scan compatible."""
        new_state = sim.step(carry_state)
        return new_state, None
    
    if sim.visualise: # only works once
        state = sim.simulate(state, steps=NUM_SIM_STEPS)
    else:
        state, _ = jax.lax.scan(simulation_step, state, None, length=NUM_SIM_STEPS)

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


def compute_loss(params, grid_x, grid_y, offset_x, offset_y, chord, sim):
    loss, _ = compute_loss_with_aux(params, grid_x, grid_y, offset_x, offset_y, chord, sim)
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

    # Create simulator once (without SDF, we'll update solid_mask in each iteration)
    sim = FluidGrid(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        cell_size=CELL_SIZE,
        dt=0.05,
        diffusion=0.001,
        boundary_type=2,
        visualise=False, # only works once then will crash -> for debugging
        show_velocity=True,
        show_cell_property="curl",
        show_cell_centered_velocity=True,
    )

    def loss_fn(p):
        return compute_loss(p, grid_x, grid_y, offset_x, offset_y, CHORD, sim)

    def aux_fn(p):
        return compute_loss_with_aux(p, grid_x, grid_y, offset_x, offset_y, CHORD, sim)

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
