# """
# Test script to verify airfoil integration in FluidGrid
# """
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# from turbodiff.core.fluid_grid import FluidGrid
# from turbodiff.utils.sdf_generator import prepare_geometry

# def test_airfoil_integration():
#     # Grid dimensions
#     H, W = 200, 400
    
#     # Prepare airfoil geometry
#     grid_sdf, path, kdtree = prepare_geometry(
#         grid_height=H,
#         grid_width=W,
#         dat_filepath="Airfoils\\RAE2822.dat",
#         chord_length=120,
#         offset_x=50,
#         offset_y=100  # Moved airfoil down a bit
#     )
    
#     # Create fluid grid with airfoil
#     fluid_grid = FluidGrid(
#         height=H,
#         width=W,
#         cell_size=1.0,
#         diffusion=0.01,
#         viscosity=0.01,
#         dt=0.01,
#         sources=[(10, 50, 1.0), (10, 100, 1.0), (10, 150, 1.0)],  # Add some dye sources on the left
#         field_type="wind tunnel",  # Use wind tunnel flow
#         visualise=True,
#         show_cell_property="density",
#         show_velocity=True,
#         grid_sdf=grid_sdf,
#         path=path,
#         kdtree=kdtree
#     )
    
#     print(f"Created grid with {H}x{W} cells")
    
#     # Count solid cells to verify airfoil is there
#     solid_count = 0
#     for i in range(H):
#         for j in range(W):
#             if fluid_grid.grid[i][j].is_solid:
#                 solid_count += 1
    
#     print(f"Number of solid cells (including boundaries and airfoil): {solid_count}")
    
#     # Run simulation
#     fluid_grid.simulate(steps=1000)  # Run for 1000 steps or until window is closed

# if __name__ == "__main__":
#     test_airfoil_integration()