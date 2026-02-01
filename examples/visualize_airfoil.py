"""
Example: Visualize flow around a real airfoil from .dat file.

Interactive pygame visualization of fluid simulation
around an airfoil loaded from a standard .dat file.

Controls:
  - Hold 'A' and drag mouse to add velocity
  - Click to see cell coordinates
  - Close window to exit
"""

import os
from turbodiff import FluidGrid
from turbodiff.utils.sdf_generator import create_sdf_function


def main():
    print("Creating JAX-based fluid simulator with RAE2822 airfoil...")

    height = 80
    width = 200

    # Airfoil positioning (in grid indices)
    chord_length = 40  # pixels
    offset_x = 30  # start x position
    offset_y = height // 2  # center vertically

    # Get the airfoil .dat file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dat_filepath = os.path.join(project_root, "Airfoils", "RAE2822.dat")

    print(f"Loading airfoil from: {dat_filepath}")
    sdf = create_sdf_function(dat_filepath, chord_length, offset_x, offset_y)

    sim = FluidGrid(
        height=height,
        width=width,
        cell_size=0.01,
        diffusion=0.01,
        viscosity=0.01,
        dt=0.01,
        boundary_type=2,
        visualise=True,
        show_velocity=True,
        show_cell_property="curl",
        show_cell_centered_velocity=True,
        sdf=sdf,
    )

    print("Setting up initial state...")
    state = sim.create_initial_state()
    state = sim.set_velocity_field(state, field_type="wind tunnel")

    print("\nControls:")
    print("  - Hold 'A' and drag mouse to add velocity")
    print("  - Click to see cell coordinates")
    print("  - Close window to exit")
    print("\nRunning simulation...")

    state = sim.simulate(state, steps=-1)

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
