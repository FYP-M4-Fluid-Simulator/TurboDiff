# from turbodiff.core.fluid_grid import FluidGrid
from turbodiff import FluidGrid


def test_base():
    grid = FluidGrid(4, 4, 0, 0, 0, [(0, 0, 0.1)])
    assert grid.length == 4
    assert grid.width == 4
