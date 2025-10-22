# import math
import random

# import numpy as np
from turbodiff.core.fluid_cell import FluidCell


class FluidGrid:
    grid = [[]]

    def __init__(
        self,
        length: int,
        width: int,
        diffusion: int,
        viscosity: int,
        dt: float,
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
                [FluidCell(0, 0, False) for _ in range(width)] for _ in range(length)
            ]  # create 2d grid with all cells at density 0 and velocity 0
        else:
            self.grid = [
                [FluidCell(0, random.random(), False) for _ in range(width)]
                for _ in range(length)
            ]  # create 2d grid with all cells at density 0 and velocity 0

    def simulate(self, steps=-1):
        step = 0
        while step != steps:
            # vel_step()
            # dens_step()
            if self.visualise():
                print("Not implemented")


if __name__ == "__main__":
    print("Hello")
