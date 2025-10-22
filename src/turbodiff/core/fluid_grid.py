import math
import random
import pygame

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
        show_velocity: bool = False,
    ):
        self.length = length
        self.width = width
        self.dt = dt

        self.visualise = visualise
        self.show_velocity = show_velocity

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
                    FluidCell(0, (random.random() - 0.5, random.random() - 0.5), False)
                    for i in range(width)
                ]
                for j in range(length)
            ]  # create 2d grid with all cells at density 0 and velocity (0, 0)

        for x, y, s in sources:
            self.grid[x][y].set_source(s)

        if self.visualise:  # if visualising in pygame then initialise
            pygame.init()
            self.cell_size = 1000 // max(self.length, self.width)
            self.screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.length * self.cell_size)
            )

    def simulate(self, steps: int = -1):
        step = 0
        while step != steps:
            if self.visualise:
                for event in pygame.event.get():  # allow exit
                    if event.type == pygame.QUIT:
                        return

            self._dens_step()
            # _vel_step()

            if self.visualise:  # move visualisation forward
                self._draw_grid()

            step += 1

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

    def _draw_grid(self):
        self.screen.fill((0, 0, 0))  # clear background

        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                # Draw cell color (density)
                val = max(0, min(255, int(cell.density * 255)))
                color = (val, val, val)
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, color, rect)

                # Show velocity as arrows - if flag set
                # Show velocity as arrows - if flag set
                if self.show_velocity:
                    vx, vy = cell.velocity
                    cx = x * self.cell_size + self.cell_size // 2
                    cy = y * self.cell_size + self.cell_size // 2

                    # Fixed arrow length for visibility
                    scale = self.cell_size * 0.4
                    end_x = cx + vx * scale
                    end_y = cy + vy * scale

                    # Compute magnitude for color mapping
                    mag = (vx**2 + vy**2) ** 0.5
                    mag = min(1.0, mag)
                    r = int(255 * mag)
                    g = 0
                    b = int(255 * (1 - mag))
                    color = (r, g, b)

                    # Draw main arrow line
                    pygame.draw.line(self.screen, color, (cx, cy), (end_x, end_y), 2)

                    angle = math.atan2(vy, vx)
                    tip_len = self.cell_size / 10  # length of arrowhead sides
                    spread = math.radians(25)  # angle between the two sides

                    # Compute the two base points of the triangle
                    left_x = end_x - tip_len * math.cos(angle - spread)
                    left_y = end_y - tip_len * math.sin(angle - spread)
                    right_x = end_x - tip_len * math.cos(angle + spread)
                    right_y = end_y - tip_len * math.sin(angle + spread)

                    pygame.draw.polygon(
                        self.screen,
                        color,
                        [(end_x, end_y), (left_x, left_y), (right_x, right_y)],
                    )

        pygame.display.flip()


if __name__ == "__main__":
    grid = FluidGrid(
        10,
        10,
        0,
        0,
        0.1,
        [(3, 3, 0.5), (5, 5, 0.2)],
        random_vel=True,
        visualise=True,
        show_velocity=True,
    )
    grid.simulate()
