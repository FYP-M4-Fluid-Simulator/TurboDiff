# import math
import random

# import numpy as np
from turbodiff.core.fluid_cell import FluidCell


class FluidGrid:
    grid: list[list[FluidCell]]

    def __init__(
        self,
        length: int,
        width: int,
        diffusion: int,
        viscosity: int,
        dt: float,
        sources: list[tuple[int, int, int]],  # list of x,y positions for sources of dye
        random_vel: bool = False,
        visualise: bool = False,
    ):
        self.length = length
        self.width = width
        self.dt = dt
        self.visualise = visualise

        self.diff = diffusion
        self.visc = viscosity

        if not random_vel:
            self.grid = [
                [FluidCell(0, (0, 0), False) for _ in range(width)]
                for _ in range(length)
            ]  # create 2d grid with all cells at density 0 and velocity (0, 0)
        else:
            self.grid = [
                [
                    FluidCell(0, (random.random(), random.random()), False)
                    for i in range(width)
                ]
                for j in range(length)
            ]  # create 2d grid with all cells at density 0 and velocity (0, 0)

        for x, y, s in sources:
            self.grid[x][y].set_source(s)

    def simulate(self, steps: int = -1):
        step = 0
        while step != steps:
            # _vel_step()
            self._dens_step()
            if self.visualise():
                print("Not implemented")

    # private methods
    def _dens_step(self):
        self._add_source()
        # self._diffuse()

    # def _vel_step(self, N: int, )

    # def _diffuse(self):
    #     se

    def _add_source(self):
        for row in self.grid:
            for cell in row:
                cell.add_source(self.dt)
                cell.update_cell()


if __name__ == "__main__":
    grid = FluidGrid(4, 4, 0, 0, 0, [(0, 0, 0.1)])
