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


def f(x, y):
    return ((x - 10) ** 2 + (y - 25) ** 2) ** 1 / 2 - 10


def main():
    # Create simulator
    print("Creating JAX-based fluid simulator...")
    sim = FluidGrid(
        height=20,
        width=100,
        cell_size=0.01,
        diffusion=0.01,
        viscosity=0.01,
        dt=0.01,
        boundary_type=2,
        visualise=True,
        show_velocity=True,
        show_cell_property="curl",
        show_cell_centered_velocity=True,
        sdf=f,
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
