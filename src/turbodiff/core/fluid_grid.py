import math
import random
import pygame

# import numpy as np
from turbodiff.core.fluid_cell import FluidCell


class FluidGrid:
    """
    MAC Grid representation: The grid is made up of cells and cell edges. Density is stored in cells. Velocity is stored on edges
    Grid size is height * Width
    dt: represents size of timestep
    sources: list of points which are a source, and then amount of flow coming from those cells
    """

    grid: list[list[FluidCell]]

    def __init__(
        self,
        height: int,
        width: int,
        diffusion: int,
        viscosity: int,
        dt: float,
        sources: list[
            tuple[int, int, float]
        ],  # list of x,y positions for sources of dye
        random_vel: bool = False,
        visualise: bool = False,
        show_velocity: bool = False,
    ):
        self.height = height
        self.width = width
        self.dt = dt

        self.visualise = visualise
        self.show_velocity = show_velocity

        self.diff = diffusion
        self.visc = viscosity

        self.grid = [
            [
                FluidCell(
                    (i, j), 0, i == 0 or i == width - 1 or j == 0 or j == height - 1
                )
                for i in range(width)
            ]
            for j in range(height)
        ]  # create 2d grid with all cells at density 0

        if not random_vel:
            self.velocities_x = [
                [0 for _ in range(self.width + 1)] for _ in range(self.height)
            ]
            self.velocities_y = [
                [0 for _ in range(self.width)] for _ in range(self.height + 1)
            ]
        else:
            self.velocities_x = [
                [random.random() - 0.5 for _ in range(self.width + 1)]
                for _ in range(self.height)
            ]
            self.velocities_y = [
                [random.random() - 0.5 for _ in range(self.width)]
                for _ in range(self.height + 1)
            ]

        for x, y, s in sources:
            self.grid[x][y].set_source(s)

        if self.visualise:  # if visualising in pygame then initialise
            pygame.init()
            self.cell_size = 1000 // max(self.height, self.width)
            self.screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
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
                if cell.is_solid:
                    val = 0
                else:
                    GRAY_VALUE = 15
                    val = max(
                        GRAY_VALUE,
                        GRAY_VALUE
                        + min(255 - GRAY_VALUE, int(cell.density * (255 - GRAY_VALUE))),
                    )
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

                    # Fixed arrow height for visibility
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
                    tip_len = self.cell_size / 10  # height of arrowhead sides
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
        show_velocity=False,
    )
    grid.simulate()
