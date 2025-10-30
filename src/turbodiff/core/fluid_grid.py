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
    velocities_x: list[list[float]]  # horizontal velocities on vertical edges
    velocities_y: list[list[float]]  # vertical velocities on horizontal edges
    
    next_velocities_x: list[list[float]]  # horizontal velocities on vertical edges
    next_velocities_y: list[list[float]]  # vertical velocities on horizontal edges

    def __init__(
        self,
        height: int,
        width: int,
        diffusion: float,
        viscosity: float,
        dt: float,
        sources: list[
            tuple[int, int, float]
        ],  # list of x,y positions for sources of dye
        field_type: str = "zero",
        visualise: bool = False,
        show_velocity: bool = False,
        show_cell_centered_velocity: bool = False,
    ):
        self.height = height
        self.width = width
        self.dt = dt

        self.visualise = visualise
        self.show_velocity = show_velocity
        self.show_cell_centered_velocity = show_cell_centered_velocity

        self.diffusion = diffusion
        self.viscosity = viscosity

        self.grid = [
            [
                FluidCell(
                    (i, j), 0, i == 0 or i == height - 1 or j == 0 or j == width - 1
                )
                for j in range(width)
            ]
            for i in range(height)
        ]  # create 2d grid with all cells at density 0
        
        # Set velocity field
        self.set_velocity_field(field_type)

        for x, y, s in sources:
            self.grid[x][y].set_source(s)

        if self.visualise:  # if visualising in pygame then initialise
            pygame.init()
            pygame.display.set_caption("Fluid Simulation")
            self.cell_size = 1000 // max(self.height, self.width)
            self.screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
            self.clock = pygame.time.Clock()

    def set_velocity_field(self, field_type: str = "zero"):
        if field_type == "zero": # Set all velocities to zero
            self.velocities_x = [
                [0.0 for _ in range(self.width + 1)] for _ in range(self.height)
            ]
            self.velocities_y = [
                [0.0 for _ in range(self.width)] for _ in range(self.height + 1)
            ]
            
        elif field_type == "random": # Set random velocities
            self.velocities_x = [
                [random.random() - 0.5 for _ in range(self.width + 1)]
                for _ in range(self.height)
            ]
            self.velocities_y = [
                [random.random() - 0.5 for _ in range(self.width)]
                for _ in range(self.height + 1)
            ]

            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i][j].is_solid:
                        vels = self.grid[i][j].get_edges_index()
                        self.velocities_x[vels[0][0]][vels[0][1]] = 0
                        self.velocities_x[vels[1][0]][vels[1][1]] = 0
                        self.velocities_y[vels[2][0]][vels[2][1]] = 0
                        self.velocities_y[vels[3][0]][vels[3][1]] = 0
        
        elif field_type == "spiral": # Set spiral/circular velocity field
            # initialize velocity lists
            self.velocities_x = [
                [0.0 for _ in range(self.width + 1)] for _ in range(self.height)
            ]
            self.velocities_y = [
                [0.0 for _ in range(self.width)] for _ in range(self.height + 1)
            ]
            
            # Add a circular/vortex flow pattern
            center_i = self.height // 2
            center_j = self.width // 2

            for i in range(self.height):
                for j in range(self.width + 1):
                    # Horizontal velocities - create circular flow
                    di = i - center_i
                    dj = j - center_j
                    dist = max(0.1, (di**2 + dj**2) ** 0.5)
                    # Circular flow: velocity perpendicular to radius
                    self.velocities_x[i][j] = -di / dist * 2.0

            for i in range(self.height + 1):
                for j in range(self.width):
                    # Vertical velocities
                    di = i - center_i
                    dj = j - center_j
                    dist = max(0.1, (di**2 + dj**2) ** 0.5)
                    self.velocities_y[i][j] = dj / dist * 2.0

            # Zero out velocities at solid boundaries
            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i][j].is_solid:
                        vels = self.grid[i][j].get_edges_index()
                        # print(vels)
                        self.velocities_x[vels[0][0]][vels[0][1]] = 0
                        self.velocities_x[vels[1][0]][vels[1][1]] = 0
                        self.velocities_y[vels[2][0]][vels[2][1]] = 0
                        self.velocities_y[vels[3][0]][vels[3][1]] = 0

    def simulate(self, steps: int = -1):
        step = 0
        while step != steps:
            if self.visualise:
                for event in pygame.event.get():  # allow exit
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.MOUSEBUTTONDOWN:  # for debugging
                        # Get mouse position (pixels)
                        mx, my = pygame.mouse.get_pos()

                        # Convert to grid coordinates
                        j = mx // self.cell_size
                        i = my // self.cell_size

                        # cell_edges = self.grid[i][j].get_edges_index()
                        print(f"Clicked cell: ({i}, {j})")
                        # self.velocities_x[cell_edges[0][0]][cell_edges[0][1]] += 0.2
                        # print(f"Corresponding Velocities Indices: ({cell_edges})")
                        # print(f"Corresponding Values:")
                        # print(self.velocities_x[cell_edges[0][0]][cell_edges[0][1]])
                        # print(self.velocities_x[cell_edges[1][0]][cell_edges[1][1]])
                        # print(self.velocities_y[cell_edges[2][0]][cell_edges[2][1]])
                        # print(self.velocities_y[cell_edges[3][0]][cell_edges[3][1]])

            self._vel_step()
            self._dens_step()

            if self.visualise:  # move visualisation forward
                self._draw_grid()
                self.clock.tick(30)

            step += 1

    # private methods
    def _update_cells(self):
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j].update_cell()

    def _dens_step(self):
        self._dens_add_source()
        self._dens_diffuse()
        self._dens_advect()

    def _vel_step(self):
        # As per Joe Stam paper -> but note that we represent velocities on edges so we have to handle it slightly differently -> more similar to Sebastian's vid
        # self._vel_add_source() -> TODO - should be partly generalisable from density work
        # self._vel_diffuse() -> TODO? - should be partly generalisable from density work -> not needed for inviscid/non-viscous (suitable simplification for air I believe) fluids so left for now
        # self._vel_project() # only uncomment if anything above is implemented
        # self._vel_advect() # -> TODO - should work directly for spiral field, so let's test?
        # self._vel_project() # remove curl -> TODO
        pass

    def _dens_diffuse(self):
        a = (
            self.dt * self.diffusion * self.height * self.width
        )  # controls rate of approach/equalisation
        for _ in range(
            20
        ):  # Iterations of Gauss Siedel -> Since matrix is sparse, we don't build it explicitly -> since we don't build it explicitly, we can't use utils method
            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    neighbors = [
                        self.grid[i + di][j + dj].next_density
                        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        if not self.grid[i + di][j + dj].is_solid
                    ]
                    self.grid[i][j].next_density = (
                        self.grid[i][j].density + a * (sum(neighbors))
                    ) / (1 + len(neighbors) * a)
        self._update_cells()

    def _dens_advect(self):
        """
        Advection using semi-Lagrangian method with bilinear interpolation.
        Follows particles backward in time through the velocity field.
        Uses face-centered velocities (MAC grid).
        """
        N = (
            max(self.height, self.width) - 2
        )  # Internal grid size (excluding boundaries)
        dt0 = self.dt * N

        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                # Get velocity at cell center
                x = j + 0.5
                y = i + 0.5
                u, v = self._get_velocity_at(x, y)

                # Trace particle backward in time
                x -= dt0 * u
                y -= dt0 * v

                # Clamp to grid boundaries
                x = max(0.5, min(self.width - 1.5, x))
                y = max(0.5, min(self.height - 1.5, y))
                
                # Adjust for cell centres being at (i+0.5, j+0.5)
                x -= 0.5
                y -= 0.5

                # Get integer and fractional parts for bilinear interpolation
                i0 = int(y) 
                i1 = i0 + 1
                j0 = int(x) 
                j1 = j0 + 1

                # Interpolation weights
                s1 = x - j0 
                s0 = 1 - s1
                t1 = y - i0
                t0 = 1 - t1

                # Bilinear interpolation
                self.grid[i][j].next_density = s0 * (
                    t0 * self.grid[i0][j0].density
                    + t1 * self.grid[i1][j0].density
                ) + s1 * (
                    t0 * self.grid[i0][j1].density
                    + t1 * self.grid[i1][j1].density
                )

        # Update cells and apply boundary conditions
        self._dens_set_bnd()
        self._update_cells()

    def _dens_set_bnd(self):
        """
        Apply boundary conditions for density.
        Sets boundary values based on adjacent interior values.
        """
        # Left and right boundaries
        for i in range(1, self.height - 1):
            self.grid[i][0].next_density = self.grid[i][1].next_density
            self.grid[i][self.width - 1].next_density = self.grid[i][self.width - 2].next_density

        # Top and bottom boundaries
        for j in range(1, self.width - 1):
            self.grid[0][j].next_density = self.grid[1][j].next_density
            self.grid[self.height - 1][j].next_density = self.grid[self.height - 2][
                j
            ].next_density

        # Corners
        self.grid[0][0].next_density = 0.5 * (
            self.grid[1][0].next_density + self.grid[0][1].next_density
        )
        self.grid[0][self.width - 1].next_density = 0.5 * (
            self.grid[1][self.width - 1].next_density + self.grid[0][self.width - 2].next_density
        )
        self.grid[self.height - 1][0].next_density = 0.5 * (
            self.grid[self.height - 2][0].next_density
            + self.grid[self.height - 1][1].next_density
        )
        self.grid[self.height - 1][self.width - 1].next_density = 0.5 * (
            self.grid[self.height - 2][self.width - 1].next_density
            + self.grid[self.height - 1][self.width - 2].next_density
        )

    def _dens_add_source(self):
        for row in self.grid:
            for cell in row:
                cell.add_source(self.dt)
        self._update_cells()

    def _vel_advect(self):
        """
        Advection of velocities using semi-Lagrangian method with bilinear interpolation.
        Follows particles backward in time through the velocity field.
        Uses face-centered velocities (MAC grid).
        """
        N = (
            max(self.height, self.width) - 2
        )  # Internal grid size (excluding boundaries)
        dt0 = self.dt * N

        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                # Get velocity at cell center
                x = j + 0.5
                y = i + 0.5
                u, v = self._get_velocity_at(x, y)

                # Trace particle backward in time
                x -= dt0 * u
                y -= dt0 * v

                # Clamp to grid boundaries
                x = max(0.5, min(self.width - 1.5, x))
                y = max(0.5, min(self.height - 1.5, y))
                
                # Adjust for cell centres being at (i+0.5, j+0.5)
                x -= 0.5
                y -= 0.5

                # Get integer and fractional parts for bilinear interpolation
                i0 = int(y) 
                i1 = i0 + 1
                j0 = int(x) 
                j1 = j0 + 1

                # Interpolation weights
                s1 = x - j0 
                s0 = 1 - s1
                t1 = y - i0
                t0 = 1 - t1

                # Bilinear interpolation
                self.grid[i][j].next_density = s0 * (
                    t0 * self.grid[i0][j0].density
                    + t1 * self.grid[i1][j0].density
                ) + s1 * (
                    t0 * self.grid[i0][j1].density
                    + t1 * self.grid[i1][j1].density
                )
    
    def _vel_project(self):
        pass
    
    def _draw_grid(self):
        self.screen.fill((0, 0, 0))  # clear background

        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                # Draw cell color (density)
                if cell.is_solid:
                    val = 0
                else:
                    GRAY_VALUE = 30
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
        if self.show_velocity:
            if self.show_cell_centered_velocity:
                # Cell Centered Velocity
                for i in range(self.height):
                    for j in range(self.width):
                        # Get velocity at cell center
                        x = j + 0.5
                        y = i + 0.5
                        u, v = self._get_velocity_at(x, y)

                        # Get and clamp magnitude/direction
                        mag_dir_x = u
                        mag_dir_y = v
                        mag = (mag_dir_x**2 + mag_dir_y**2) ** 0.5
                        if mag != 0:
                            mag_dir_x /= mag
                            mag_dir_y /= mag
                        mag = min(1.0, mag)

                        # Map color based on magnitude
                        r = int(255 * mag)
                        g = 0
                        b = int(255 * (1 - mag))
                        color = (r, g, b)
                        # Scale arrow as per grid size for visibility
                        scale = self.cell_size * 0.4

                        # Arrow end points
                        start_x = (j + 0.5) * self.cell_size
                        start_y = (i + 0.5) * self.cell_size
                        end_x = start_x + scale * mag_dir_x
                        end_y = start_y + scale * mag_dir_y

                        pygame.draw.aaline(
                            self.screen, color, (start_x, start_y), (end_x, end_y), 2
                        )
                        
                        # Draw main arrow line
                        angle = math.atan2(mag_dir_y, mag_dir_x)
                        tip_len = scale * mag / 2  # height of arrowhead sides
                        spread = math.radians(25)  # angle between the two sides
                        
                        # compute the two base points of the triangle
                        left_x = end_x - tip_len * math.cos(angle - spread)
                        left_y = end_y - tip_len * math.sin(angle - spread)
                        right_x = end_x - tip_len * math.cos(angle + spread)
                        right_y = end_y - tip_len * math.sin(angle + spread)
                        
                        pygame.draw.aalines(
                            self.screen,
                            color,
                            points=[(end_x, end_y), (left_x, left_y), (right_x, right_y)],
                            closed=True
                        )
                        
            elif self.show_cell_centered_velocity == False:
                # Velocity for each face
                # Horizontal Velocity
                for i in range(self.height):
                    for j in range(self.width + 1):
                        # Get and clamp velocity
                        mag_dir = self.velocities_x[i][j]
                        mag_dir = min(1.0, mag_dir) if mag_dir > 0 else max(-1.0, mag_dir)

                        # Map color based on magnitude
                        r = int(255 * abs(mag_dir))
                        g = 0
                        b = int(255 * (1 - abs(mag_dir)))
                        color = (r, g, b)
                        # Scale arrow as per grid size for visibility
                        scale = self.cell_size * 0.4

                        # Arrow end points
                        start_x = j * self.cell_size
                        end_x = start_x + scale * mag_dir
                        y = (i + 0.5) * self.cell_size

                        pygame.draw.aaline(self.screen, color, (start_x, y), (end_x, y), 2)

                        # Draw main arrow line
                        angle = 0 if mag_dir > 0 else math.pi
                        tip_len = scale * abs(mag_dir) / 2  # height of arrowhead sides
                        spread = math.radians(25)  # angle between the two sides

                        # # Compute the two base points of the triangle
                        left_x = end_x - tip_len * math.cos(angle - spread)
                        left_y = y - tip_len * math.sin(angle - spread)
                        right_x = end_x - tip_len * math.cos(angle + spread)
                        right_y = y - tip_len * math.sin(angle + spread)
                        # print(left_y - y)
                        # print(y - right_y)

                        pygame.draw.aalines(
                            self.screen,
                            color,
                            points=[(end_x, y), (left_x, left_y), (right_x, right_y)],
                            closed=True
                        )

                # Vertical Velocity
                for i in range(self.height + 1):
                    for j in range(self.width):
                        # Get and clamp velocity
                        mag_dir = self.velocities_y[i][j]
                        mag_dir = min(1.0, mag_dir) if mag_dir > 0 else max(-1.0, mag_dir)

                        # Map color based on magnitude
                        r = int(255 * abs(mag_dir))
                        g = 0
                        b = int(255 * (1 - abs(mag_dir)))
                        color = (r, g, b)
                        # Scale arrow as per grid size for visibility
                        scale = self.cell_size * 0.4

                        # Arrow end points
                        x = (j + 0.5) * self.cell_size
                        start_y = i * self.cell_size
                        end_y = start_y + scale * mag_dir

                        pygame.draw.aaline(self.screen, color, (x, start_y), (x, end_y), 2)

                        # Draw main arrow line
                        angle = math.pi / 2 if mag_dir > 0 else -math.pi / 2
                        tip_len = scale * abs(mag_dir) / 2  # height of arrowhead sides
                        spread = math.radians(25)  # angle between the two sides

                        # # Compute the two base points of the triangle
                        left_x = x - tip_len * math.cos(angle - spread)
                        left_y = end_y - tip_len * math.sin(angle - spread)
                        right_x = x - tip_len * math.cos(angle + spread)
                        right_y = end_y - tip_len * math.sin(angle + spread)
                        # print(left_y - y)
                        # print(y - right_y)

                        pygame.draw.aalines(
                            self.screen,
                            color,
                            points=[(x, end_y), (left_x, left_y), (right_x, right_y)],
                            closed=True
                        )

        pygame.display.flip()

    def _sample_bilinear(self, values, px, py, ny, nx):
        px = max(0.5, min(px, nx - 1.5))
        left = int(px)
        if left > nx - 2:
            left = nx - 2
        xfrac = px - left

        py = max(0.5, min(py, ny - 1.5))
        top = int(py)
        if top > ny - 2:
            top = ny - 2
        yfrac = py - top

        right = left + 1
        bottom = top + 1

        value_top = (1 - xfrac) * values[top][left] + xfrac * values[top][right]
        value_bottom = (1 - xfrac) * values[bottom][left] + xfrac * values[bottom][
            right
        ]

        return (1 - yfrac) * value_top + yfrac * value_bottom

    def _get_velocity_at(self, px, py):
        if self.grid[int(py)][int(px)].is_solid:
            return 0.0, 0.0
        u = self._sample_bilinear(
            self.velocities_x, px, py - 0.5, self.height, self.width + 1
        )
        v = self._sample_bilinear(
            self.velocities_y, px - 0.5, py, self.height + 1, self.width
        )
        return u, v


if __name__ == "__main__":
    grid = FluidGrid(
        height=50,
        width=50,
        diffusion=0.01,
        viscosity=0,
        dt=0.02,
        sources=[(10, 25, 100)],
        field_type="spiral",
        visualise=True,
        show_velocity=True,
        show_cell_centered_velocity=True,
    )

    grid.simulate()
