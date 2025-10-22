from turbodiff.core.fluid_grid import FluidGrid


def test_tester():
    Grid = FluidGrid(1, 1, 0, 0, 0)
    assert Grid.length == 1
    assert Grid.width == 1
