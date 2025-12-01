"""
Example: Visualize fluid simulation with pygame.

This script demonstrates our fluid simulator
with interactive visualization using pygame.

Controls:
  - Hold 'A' and drag mouse to add velocity
  - Click to see cell coordinates
  - Close window to exit
"""

from turbodiff import FluidGrid


def main():
    # Create simulator
    print("Creating JAX-based fluid simulator...")
    sim = FluidGrid(
        height=50,
        width=50,
        cell_size=0.01,
        diffusion=0.001,
        viscosity=0.001,
        dt=0.01,
        visualise=True,
        show_velocity=True,
        show_cell_centered_velocity=True,
    )

    # Create initial state
    print("Setting up initial state...")
    state = sim.create_initial_state()

    # Set spiral velocity field
    state = sim.set_velocity_field(state, field_type="wind tunnel")

    # Add density source at (10, 10)
    state = sim.set_sources(state, [(10, 10, 300.0)])

    print("\nControls:")
    print("  - Hold 'A' and drag mouse to add velocity")
    print("  - Click to see cell coordinates")
    print("  - Close window to exit")
    print("\nRunning simulation...")

    # Run simulation with built-in visualization
    state = sim.simulate(state, steps=-1)

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
