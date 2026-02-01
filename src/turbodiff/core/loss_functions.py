"""
Loss functions for fluid-based shape optimization.

This module provides physics-based loss functions for computing
drag, lift, and other aerodynamic metrics from fluid simulation results.
"""

import jax
import jax.numpy as jnp
from jax import jit
from turbodiff.core.types import Array
from turbodiff.core.fluid_grid_jax import FluidState


def drag_loss(state: FluidState, obstacle_mask: Array) -> float:
    """
    Compute drag force on obstacle by integrating pressure on obstacle surfaces.

    Drag is computed as the pressure difference between front (windward) and
    back (leeward) surfaces of the obstacle in the flow direction.

    For smooth masks (float 0-1), we use gradients to identify surface regions.

    Args:
        state: Fluid simulation state
        obstacle_mask: Mask marking obstacle cells (float 0-1 for differentiability)

    Returns:
        Drag force (scalar)
    """
    pressure = state.pressure.values

    # For smooth masks, use mask gradient to identify surfaces
    # Front surface: mask increases in x-direction (flow enters obstacle)
    # Back surface: mask decreases in x-direction (flow exits obstacle)

    # Compute mask gradient in x-direction (flow direction)
    mask_grad_x = obstacle_mask[:, 1:] - obstacle_mask[:, :-1]

    # Front surface weight: positive gradient (entering obstacle)
    # Use smooth approximation for differentiability
    front_surface = jax.nn.sigmoid(mask_grad_x * 10.0)

    # Back surface weight: negative gradient (leaving obstacle)
    back_surface = jax.nn.sigmoid(-mask_grad_x * 10.0)

    # Pressure on front faces (left side of each cell boundary)
    p_front = pressure[:, :-1]
    # Pressure on back faces (right side of each cell boundary)
    p_back = pressure[:, 1:]

    # Integrate pressure force on surfaces
    # Drag = ∫(p_front) dA - ∫(p_back) dA
    drag_front = jnp.sum(p_front * front_surface)
    drag_back = jnp.sum(p_back * back_surface)

    # Total drag (can be negative if back pressure > front pressure)
    # For optimization, we want to minimize absolute drag magnitude
    drag = drag_front - drag_back

    # Return absolute value to ensure drag is always positive
    return jnp.abs(drag)


def lift_loss(state: FluidState, obstacle_mask: Array) -> float:
    """
    Compute lift force on obstacle from pressure gradients.

    Approximates lift by integrating pressure differences on the
    top and bottom surfaces of the obstacle.

    Args:
        state: Fluid simulation state
        obstacle_mask: Boolean mask marking obstacle cells

    Returns:
        Lift force magnitude (scalar)
    """
    pressure = state.pressure.values

    # Compute pressure gradient in y-direction (perpendicular to flow)
    pressure_grad_y = pressure[1:, :] - pressure[:-1, :]

    # Identify obstacle boundaries
    top_boundary = (~obstacle_mask[:-1, :]) & obstacle_mask[1:, :]
    bottom_boundary = obstacle_mask[:-1, :] & (~obstacle_mask[1:, :])

    # Integrate pressure forces
    lift_top = jnp.sum(jnp.where(top_boundary, pressure_grad_y, 0.0))
    lift_bottom = jnp.sum(jnp.where(bottom_boundary, -pressure_grad_y, 0.0))

    total_lift = lift_top + lift_bottom
    return total_lift


def lift_to_drag_ratio(
    state: FluidState, obstacle_mask: Array, epsilon: float = 1e-6
) -> float:
    """
    Compute lift-to-drag ratio (L/D) for airfoil performance.

    Returns negative L/D for minimization (maximize L/D by minimizing -L/D).

    Args:
        state: Fluid simulation state
        obstacle_mask: Boolean mask marking obstacle cells
        epsilon: Small value to avoid division by zero

    Returns:
        Negative L/D ratio
    """
    lift = lift_loss(state, obstacle_mask)
    drag = drag_loss(state, obstacle_mask)

    return -lift / jnp.maximum(jnp.abs(drag), epsilon)


# ============================================================================
# Velocity-Based Metrics
# ============================================================================


def velocity_magnitude_loss(state: FluidState, weight_mask: Array = None) -> float:
    """
    Compute kinetic energy in the flow field.

    Useful for maximizing flow speed or minimizing flow disruption.

    Args:
        state: Fluid simulation state
        weight_mask: Optional mask to weight different regions

    Returns:
        Mean velocity magnitude squared
    """
    u = state.velocity.u
    v = state.velocity.v

    # Average velocities to cell centers
    u_center = (u[:, :-1] + u[:, 1:]) / 2
    v_center = (v[:-1, :] + v[1:, :]) / 2

    # Velocity magnitude squared
    vel_mag_sq = u_center**2 + v_center**2

    if weight_mask is not None:
        vel_mag_sq = vel_mag_sq * weight_mask

    return jnp.mean(vel_mag_sq)


@jit
def turbulent_kinetic_energy_loss(state: FluidState) -> float:
    """
    Estimate turbulent kinetic energy from velocity gradients.

    Higher values indicate more turbulence (typically undesirable).

    Args:
        state: Fluid simulation state

    Returns:
        TKE estimate (scalar)
    """
    u = state.velocity.u
    v = state.velocity.v

    # Velocity gradients as proxy for fluctuations
    du_dx = u[:, 1:] - u[:, :-1]
    dv_dy = v[1:, :] - v[:-1, :]

    tke = jnp.mean(du_dx**2 + dv_dy**2)
    return tke


@jit
def pressure_drop_loss(
    state: FluidState, inlet_mask: Array, outlet_mask: Array
) -> float:
    """
    Compute pressure drop between inlet and outlet regions.

    Useful for minimizing flow resistance in channels (topology optimization).
    This matches PhiFlow's implementation: loss = sum(p * (inlet_mask - outlet_mask))

    Args:
        state: Fluid simulation state
        inlet_mask: Mask marking inlet region (float 0-1)
        outlet_mask: Mask marking outlet region (float 0-1)

    Returns:
        Pressure difference: sum(p_inlet) - sum(p_outlet)
    """
    pressure = state.pressure.values

    # PhiFlow-style: sum(p * (inlet_mask - outlet_mask))
    # Equivalent to: sum(p * inlet_mask) - sum(p * outlet_mask)
    pressure_drop = jnp.sum(pressure * (inlet_mask - outlet_mask))

    return pressure_drop


# ============================================================================
# Airfoil Geometric Constraints
# ============================================================================


def thickness_constraint_loss(
    thickness: Array,
    min_thickness: float = 0.08,
    max_thickness: float = 0.22,
    weight: float = 500.0,
) -> float:
    """
    Penalize airfoil thickness outside allowed range.

    Args:
        thickness: Thickness values at each chordwise location
        min_thickness: Minimum allowed thickness
        max_thickness: Maximum allowed thickness
        weight: Penalty weight

    Returns:
        Thickness constraint loss (0 if within bounds)
    """
    max_t = jnp.max(thickness)

    # Penalize if too thin
    too_thin = jnp.maximum(0.0, min_thickness - max_t)

    # Penalize if too thick
    too_thick = jnp.maximum(0.0, max_t - max_thickness)

    return (too_thin + too_thick) * weight


def crossover_validity_loss(
    y_upper: Array,
    y_lower: Array,
    weight: float = 1000.0,
) -> float:
    """
    Penalize invalid airfoil geometry where upper surface crosses below lower.

    Args:
        y_upper: Upper surface y-coordinates
        y_lower: Lower surface y-coordinates
        weight: Penalty weight

    Returns:
        Validity loss (0 if valid geometry)
    """
    # Thickness at each point
    thickness = y_upper - y_lower

    # Penalize negative thickness (crossover)
    crossover_violation = jnp.maximum(0.0, -thickness)

    return jnp.sum(crossover_violation) * weight


def airfoil_combined_loss(
    C_L: float,
    C_D: float,
    geo_loss: float,
    w_lift: float = 1000.0,
    w_drag: float = 100.0,
    w_ratio: float = 100.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Combined loss for airfoil optimization.

    Maximizes lift, minimizes drag, maximizes L/D ratio,
    while respecting geometric constraints.

    Args:
        C_L: Lift coefficient
        C_D: Drag coefficient
        geo_loss: Geometric constraint loss
        w_lift: Weight for lift (negative because we maximize)
        w_drag: Weight for drag magnitude
        w_ratio: Weight for L/D ratio (negative because we maximize)
        epsilon: Small value to avoid division by zero

    Returns:
        Combined loss for minimization
    """
    # Maximize lift -> minimize negative lift
    loss_lift = -C_L * w_lift

    # Minimize drag magnitude
    loss_drag = jnp.abs(C_D) * w_drag

    # Maximize L/D -> minimize -L/D
    C_D_safe = jnp.maximum(jnp.abs(C_D), epsilon)
    loss_ratio = -(C_L / C_D_safe) * w_ratio

    return loss_lift + loss_drag + loss_ratio + geo_loss


__all__ = [
    "drag_loss",
    "lift_loss",
    "lift_to_drag_ratio",
    "velocity_magnitude_loss",
    "turbulent_kinetic_energy_loss",
    "pressure_drop_loss",
    "thickness_constraint_loss",
    "crossover_validity_loss",
    "airfoil_combined_loss",
]
