"""
Example: Visualize flow around an airfoil defined by CST weights.

Interactive pygame visualization of fluid simulation
around an airfoil generated from CST parameters.

Controls:
  - Hold 'A' and drag mouse to add velocity
  - Click to see cell coordinates
  - Close window to exit
"""

import jax.numpy as jnp
from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.core.airfoil_optimization import create_airfoil_solid_mask


def main():
    print("Creating JAX-based fluid simulator with CST airfoil...")

    height = 80
    width = 200

    # Simulation parameters
    cell_size = 0.01
    dt = 0.01

    # Airfoil positioning (physical units)
    chord_length = 40 * cell_size
    offset_x = 30 * cell_size
    offset_y = (height // 2) * cell_size

    # Grid coordinates for mask generation
    j_coords = jnp.arange(width, dtype=jnp.float32)
    i_coords = jnp.arange(height, dtype=jnp.float32)
    j_grid, i_grid = jnp.meshgrid(j_coords, i_coords, indexing="xy")
    grid_x = (j_grid + 0.5) * cell_size
    grid_y = (i_grid + 0.5) * cell_size

    # RAE2822-like CST weights
    weights_upper = jnp.array([0.18, 0.22, 0.20, 0.18, 0.15, 0.12])
    weights_lower = jnp.array([-0.10, -0.08, -0.06, -0.05, -0.04, -0.03])

    print("Generating airfoil mask from CST weights...")
    solid_mask = create_airfoil_solid_mask(
        weights_upper,
        weights_lower,
        grid_x,
        grid_y,
        offset_x,
        offset_y,
        chord=chord_length,
        num_cst_points=200,
        sharpness=50.0,
    )

    # Threshold mask to ensure clear visualization
    solid_mask = jnp.where(solid_mask < 0.05, 0.0, solid_mask)

    # Initialize FluidGrid
    sim = FluidGrid(
        height=height,
        width=width,
        cell_size=cell_size,
        diffusion=0.01,
        viscosity=0.01,
        dt=dt,
        boundary_type=2,  # No right boundary
        visualise=True,
        show_velocity=True,
        show_cell_property="curl",
        show_cell_centered_velocity=True,
        sdf=None,
    )

    print("Setting up initial state...")
    state = sim.create_initial_state()

    # Combine boundary mask with airfoil mask
    combined_mask = jnp.maximum(sim.solid_mask, solid_mask)
    sim.solid_mask = combined_mask

    state = FluidState(
        density=state.density,
        velocity=state.velocity,
        pressure=state.pressure,
        solid_mask=combined_mask,
        sources=state.sources,
        time=state.time,
        step=state.step,
    )

    state = sim.set_velocity_field(state, field_type="wind tunnel")

    # Add smoke sources at the inlet
    source_positions = []
    for i in range(height):
        if i % 8 < 4:
            source_positions.append((i, 5, 2.0))

    state = sim.set_sources(state, source_positions)

    print("\nControls:")
    print("  - Hold 'A' and drag mouse to add velocity")
    print("  - Click to see cell coordinates")
    print("  - Close window to exit")
    print("\nRunning simulation...")

    try:
        sim.simulate(state, steps=-1)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
