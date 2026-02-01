"""
Airfoil shape optimization using differentiable CFD.

This module provides a complete workflow for optimizing airfoil shapes
using CST parametrization, soft masking, and gradient-based optimization
through the FluidGrid simulator.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import jax.numpy as jnp
from jax import Array, grad

from turbodiff.core.airfoil import (
    generate_cst_coords,
    thickness_at_x,
)
from turbodiff.core.masking import (
    create_airfoil_mask,
    interpolate_surface_to_grid,
)
from turbodiff.core.loss_functions import (
    thickness_constraint_loss,
    crossover_validity_loss,
    airfoil_combined_loss,
)
from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState


@dataclass
class AirfoilOptConfig:
    """Configuration for airfoil optimization."""

    # Grid settings
    height: int = 64
    width: int = 128
    cell_size: float = 0.1

    # Airfoil placement (in grid coordinates)
    airfoil_offset_x: float = 2.0  # Distance from inlet
    airfoil_offset_y: float = 3.2  # Vertical center (height/2 * cell_size)
    chord_length: float = 1.0

    # Simulation settings
    dt: float = 0.05
    num_steps: int = 100
    reynolds: float = 100.0
    diffusion: float = 0.001

    # Inflow settings
    inflow_velocity: float = 1.0
    angle_of_attack_deg: float = 5.0

    # Optimization settings
    learning_rate: float = 0.01
    max_epochs: int = 50
    grad_clip: float = 0.1

    # Geometric constraints
    min_thickness: float = 0.08
    max_thickness: float = 0.22

    # CST settings
    n_order: int = 5  # Bernstein polynomial order


def compute_grid_coordinates(
    height: int,
    width: int,
    cell_size: float,
) -> Tuple[Array, Array]:
    """
    Compute grid cell center coordinates.

    Args:
        height: Grid height in cells
        width: Grid width in cells
        cell_size: Size of each cell

    Returns:
        (grid_x, grid_y) arrays of shape (height, width)
    """
    j = jnp.arange(width)
    i = jnp.arange(height)
    jj, ii = jnp.meshgrid(j, i, indexing="xy")

    # Cell centers
    grid_x = (jj + 0.5) * cell_size
    grid_y = (ii + 0.5) * cell_size

    return grid_x, grid_y


def create_airfoil_solid_mask(
    weights_upper: Array,
    weights_lower: Array,
    grid_x: Array,
    grid_y: Array,
    offset_x: float,
    offset_y: float,
    chord: float = 1.0,
    num_cst_points: int = 100,
    sharpness: float = 50.0,
) -> Array:
    """
    Create soft solid mask for airfoil from CST parameters.

    Args:
        weights_upper: Upper surface CST weights
        weights_lower: Lower surface CST weights
        grid_x: X-coordinates of grid, shape (H, W)
        grid_y: Y-coordinates of grid, shape (H, W)
        offset_x: Airfoil leading edge x-position
        offset_y: Airfoil centerline y-position
        chord: Chord length
        num_cst_points: Points for CST curve
        sharpness: Sigmoid sharpness for mask

    Returns:
        Soft mask, shape (H, W), ~1 inside airfoil, ~0 outside
    """
    # Generate CST airfoil coordinates
    x_cst, y_upper_cst, y_lower_cst = generate_cst_coords(
        weights_upper, weights_lower, num_points=num_cst_points
    )

    # Scale to chord length and offset
    x_cst_world = x_cst * chord + offset_x
    y_upper_cst_world = y_upper_cst * chord + offset_y
    y_lower_cst_world = y_lower_cst * chord + offset_y

    # Interpolate surfaces to grid
    y_upper_at_grid = interpolate_surface_to_grid(
        grid_x, x_cst_world, y_upper_cst_world
    )
    y_lower_at_grid = interpolate_surface_to_grid(
        grid_x, x_cst_world, y_lower_cst_world
    )

    # Create soft mask
    mask = create_airfoil_mask(
        grid_x,
        grid_y,
        y_upper_at_grid,
        y_lower_at_grid,
        x_min=offset_x,
        x_max=offset_x + chord,
        sharpness=sharpness,
    )

    return mask


def compute_geometric_loss(
    weights_upper: Array,
    weights_lower: Array,
    min_thickness: float = 0.08,
    max_thickness: float = 0.22,
) -> float:
    """
    Compute geometric constraint loss for airfoil.

    Args:
        weights_upper: Upper surface CST weights
        weights_lower: Lower surface CST weights
        min_thickness: Minimum allowed thickness
        max_thickness: Maximum allowed thickness

    Returns:
        Total geometric constraint loss
    """
    # Generate airfoil coordinates
    x, y_upper, y_lower = generate_cst_coords(weights_upper, weights_lower)

    # Thickness distribution
    thickness = thickness_at_x(y_upper, y_lower)

    # Validity loss (no crossover)
    geo_loss = crossover_validity_loss(y_upper, y_lower)

    # Thickness bounds loss
    geo_loss = geo_loss + thickness_constraint_loss(
        thickness, min_thickness, max_thickness
    )

    return geo_loss


def compute_force_coefficients(
    state: FluidState,
    obstacle_mask: Array,
    inflow_velocity: float = 1.0,
    chord_length: float = 1.0,
    fluid_density: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute lift and drag coefficients from simulation state.

    Uses pressure gradients on obstacle surface.

    Args:
        state: Current fluid simulation state
        obstacle_mask: Soft obstacle mask
        inflow_velocity: Reference velocity
        chord_length: Reference length
        fluid_density: Fluid density

    Returns:
        (C_L, C_D) lift and drag coefficients
    """
    pressure = state.pressure.values

    # Surface normals from mask gradient
    mask_grad_x = obstacle_mask[:, 1:] - obstacle_mask[:, :-1]
    mask_grad_y = obstacle_mask[1:, :] - obstacle_mask[:-1, :]

    # Pressure on shifted grid
    p_x = (pressure[:, :-1] + pressure[:, 1:]) / 2
    p_y = (pressure[:-1, :] + pressure[1:, :]) / 2

    # Force integration
    F_x = -jnp.sum(p_x * mask_grad_x)
    F_y = -jnp.sum(p_y * mask_grad_y)

    # Normalize to coefficients
    q = 0.5 * fluid_density * inflow_velocity**2
    C_D = F_x / (q * chord_length)
    C_L = F_y / (q * chord_length)

    return C_L, C_D


def run_simulation(
    simulator: FluidGrid,
    state: FluidState,
    num_steps: int,
) -> FluidState:
    """
    Run fluid simulation for specified number of steps.

    Args:
        simulator: FluidGrid instance
        state: Initial state
        num_steps: Number of time steps

    Returns:
        Final simulation state
    """
    for _ in range(num_steps):
        state = simulator.step(state)
    return state


def objective_function(
    params: Array,
    config: AirfoilOptConfig,
    simulator: FluidGrid,
    grid_x: Array,
    grid_y: Array,
) -> Tuple[float, Dict[str, Any]]:
    """
    Full objective function for airfoil optimization.

    Args:
        params: Concatenated CST weights [upper, lower]
        config: Optimization configuration
        simulator: FluidGrid simulator
        grid_x: Grid x-coordinates
        grid_y: Grid y-coordinates

    Returns:
        (loss, aux_data) where aux_data contains C_L, C_D, geo_loss
    """
    n_weights = config.n_order + 1
    weights_upper = params[:n_weights]
    weights_lower = params[n_weights:]

    # 1. Compute geometric loss
    geo_loss = compute_geometric_loss(
        weights_upper,
        weights_lower,
        config.min_thickness,
        config.max_thickness,
    )

    # 2. Create airfoil mask
    obstacle_mask = create_airfoil_solid_mask(
        weights_upper,
        weights_lower,
        grid_x,
        grid_y,
        config.airfoil_offset_x,
        config.airfoil_offset_y,
        config.chord_length,
    )

    # 3. Create initial state with airfoil mask
    state = simulator.create_initial_state()

    # Update solid mask in state (for differentiability)
    state = FluidState(
        density=state.density,
        velocity=state.velocity,
        pressure=state.pressure,
        solid_mask=obstacle_mask,
        sources=state.sources,
        time=state.time,
        step=state.step,
    )

    # 4. Run simulation
    final_state = run_simulation(simulator, state, config.num_steps)

    # 5. Compute aerodynamic coefficients
    C_L, C_D = compute_force_coefficients(
        final_state,
        obstacle_mask,
        config.inflow_velocity,
        config.chord_length,
    )

    # 6. Compute combined loss
    total_loss = airfoil_combined_loss(C_L, C_D, geo_loss)

    aux_data = {
        "C_L": C_L,
        "C_D": C_D,
        "geo_loss": geo_loss,
    }

    return total_loss, aux_data


def optimize_airfoil(
    config: Optional[AirfoilOptConfig] = None,
    initial_weights_upper: Optional[Array] = None,
    initial_weights_lower: Optional[Array] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full airfoil optimization loop.

    Args:
        config: Optimization configuration (uses defaults if None)
        initial_weights_upper: Initial upper surface weights
        initial_weights_lower: Initial lower surface weights
        verbose: Print progress

    Returns:
        Dictionary with optimized weights, history, and final metrics
    """
    if config is None:
        config = AirfoilOptConfig()

    n_weights = config.n_order + 1

    # Initialize weights
    if initial_weights_upper is None:
        initial_weights_upper = jnp.ones(n_weights) * 0.15
    if initial_weights_lower is None:
        initial_weights_lower = jnp.ones(n_weights) * (-0.10)

    params = jnp.concatenate([initial_weights_upper, initial_weights_lower])

    # Create simulator
    simulator = FluidGrid(
        height=config.height,
        width=config.width,
        cell_size=config.cell_size,
        dt=config.dt,
        diffusion=config.diffusion,
        boundary_type=2,  # Wind tunnel mode
    )

    # Compute grid coordinates
    grid_x, grid_y = compute_grid_coordinates(
        config.height, config.width, config.cell_size
    )

    # Create gradient function
    def loss_fn(p):
        loss, aux = objective_function(p, config, simulator, grid_x, grid_y)
        return loss

    grad_fn = grad(loss_fn)

    # Optimization history
    history = {
        "loss": [],
        "C_L": [],
        "C_D": [],
        "params": [params.copy()],
    }

    # Optimization loop
    learning_rate = config.learning_rate

    for epoch in range(config.max_epochs):
        # Compute loss and gradients
        loss, aux = objective_function(params, config, simulator, grid_x, grid_y)
        grads = grad_fn(params)

        # Gradient clipping
        grad_norm = jnp.linalg.norm(grads)
        if grad_norm > config.grad_clip:
            grads = grads * (config.grad_clip / grad_norm)

        # Check for NaN
        if jnp.isnan(loss) or jnp.any(jnp.isnan(grads)):
            if verbose:
                print(f"Epoch {epoch+1}: NaN detected, reducing learning rate")
            learning_rate *= 0.5
            continue

        # Update parameters
        params = params - learning_rate * grads

        # Record history
        history["loss"].append(float(loss))
        history["C_L"].append(float(aux["C_L"]))
        history["C_D"].append(float(aux["C_D"]))
        history["params"].append(params.copy())

        if verbose:
            C_L, C_D = aux["C_L"], aux["C_D"]
            L_D = C_L / max(abs(C_D), 1e-12)
            print(
                f"Epoch {epoch+1:3d} | Loss: {loss:10.4f} | C_L: {C_L:.6f} | C_D: {C_D:.6f} | L/D: {L_D:.2f}"
            )

    # Final results
    return {
        "weights_upper": params[:n_weights],
        "weights_lower": params[n_weights:],
        "history": history,
        "final_C_L": history["C_L"][-1] if history["C_L"] else 0.0,
        "final_C_D": history["C_D"][-1] if history["C_D"] else 0.0,
    }


__all__ = [
    "AirfoilOptConfig",
    "compute_grid_coordinates",
    "create_airfoil_solid_mask",
    "compute_geometric_loss",
    "compute_force_coefficients",
    "run_simulation",
    "objective_function",
    "optimize_airfoil",
]
