import math
import random
import pygame

# import numpy as np
from turbodiff.core.fluid_cell import FluidCell
from typing import Callable


class FluidGrid:
    """
    MAC Grid representation: The grid is made up of cells and cell edges. Density is stored in cells. Velocity is stored on edges
    Grid size is height * width
    dt: represents size of timestep in seconds
    sources: list of points which are a source, and then amount of flow coming from those cells
    cell size: represents height/width of each cell in metres
    velcoities stored on cell edges as metres per second
    """

    grid: list[list[FluidCell]]
    velocities_x: list[
        list[float]
    ]  # horizontal velocities on vertical edges -> positive is to right
    velocities_y: list[
        list[float]
    ]  # vertical velocities on horizontal edges -> positive is downwards

    next_velocities_x: list[list[float]]  # horizontal velocities on vertical edges
    next_velocities_y: list[list[float]]  # vertical velocities on horizontal edges

    def __init__(
        self,
        height: int,
        width: int,
        cell_size: float,
        diffusion: float,
        viscosity: float,
        dt: float,
        sources: list[
            tuple[int, int, float]
        ],  # list of x,y positions for sources of dye
        sdf: Callable[[int, int], float] | None = None,
        field_type: str = "zero",
        visualise: bool = False,
        show_cell_property: str = "density",
        show_velocity: bool = False,
        show_cell_centered_velocity: bool = False,
    ):
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.dt = dt

        self.visualise = visualise
        self.show_cell_property = show_cell_property

        self.show_velocity = show_velocity
        self.show_cell_centered_velocity = show_cell_centered_velocity

        self.diffusion = diffusion
        self.viscosity = viscosity

        self.grid = [
            [
                FluidCell(
                    (i, j),
                    0,
                    (i == 0 or i == height - 1 or j == 0 or j == width - 1)
                    or (sdf is not None and sdf(i, j) < 0),
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
            self.display_size = 1000 // max(self.height, self.width)
            self.screen = pygame.display.set_mode(
                (self.width * self.display_size, self.height * self.display_size)
            )
            self.clock = pygame.time.Clock()

    def set_velocity_field(self, field_type: str = "zero"):
        if field_type == "zero":  # Set all velocities to zero
            self.velocities_x = [
                [0.0 for _ in range(self.width + 1)] for _ in range(self.height)
            ]
            self.velocities_y = [
                [0.0 for _ in range(self.width)] for _ in range(self.height + 1)
            ]
            self.next_velocities_x = [
                [0.0 for _ in range(self.width + 1)] for _ in range(self.height)
            ]
            self.next_velocities_y = [
                [0.0 for _ in range(self.width)] for _ in range(self.height + 1)
            ]

        elif field_type == "random":  # Set random velocities
            self.velocities_x = [
                [random.random() - 0.5 for _ in range(self.width + 1)]
                for _ in range(self.height)
            ]
            self.velocities_y = [
                [random.random() - 0.5 for _ in range(self.width)]
                for _ in range(self.height + 1)
            ]
            self.next_velocities_x = [
                [0.0 for _ in range(self.width + 1)] for _ in range(self.height)
            ]
            self.next_velocities_y = [
                [0.0 for _ in range(self.width)] for _ in range(self.height + 1)
            ]

            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i][j].is_solid:
                        vels = self.grid[i][j].get_edges_index()
                        self.velocities_x[vels[0][0]][vels[0][1]] = 0
                        self.velocities_x[vels[1][0]][vels[1][1]] = 0
                        self.velocities_y[vels[2][0]][vels[2][1]] = 0
                        self.velocities_y[vels[3][0]][vels[3][1]] = 0

        elif field_type == "spiral":  # Set spiral/circular velocity field
            # initialize velocity lists
            self.velocities_x = [
                [0.0 for _ in range(self.width + 1)] for _ in range(self.height)
            ]
            self.velocities_y = [
                [0.0 for _ in range(self.width)] for _ in range(self.height + 1)
            ]
            self.next_velocities_x = [
                [0.0 for _ in range(self.width + 1)] for _ in range(self.height)
            ]
            self.next_velocities_y = [
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
                    dist = max(0.001, (di**2 + dj**2) ** 0.5)
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

        elif field_type == "wind tunnel":  # Set left to right flow
            # initialize velocity lists
            self.velocities_x = [
                [0.0 for _ in range(self.width + 1)] for _ in range(self.height)
            ]
            self.velocities_y = [
                [0.0 for _ in range(self.width)] for _ in range(self.height + 1)
            ]
            self.next_velocities_x = [
                [0.0 for _ in range(self.width + 1)] for _ in range(self.height)
            ]
            self.next_velocities_y = [
                [0.0 for _ in range(self.width)] for _ in range(self.height + 1)
            ]

            for i in range(1, self.height - 1):
                self.velocities_x[i][1] = 2.0  # constant rightward flow
                self.grid[i][-1].is_solid = False  # ensure right boundary is not solid

    def simulate(self, steps: int = -1):
        step = 0
        prev_mouse_pos = None  # Track previous mouse position for velocity calculation

        while step != steps:
            if self.visualise:
                # Check if 'A' key is being held down
                keys = pygame.key.get_pressed()
                is_painting = keys[pygame.K_a]

                # Get current mouse position
                mouse_pos = pygame.mouse.get_pos()

                # If 'A' is held and mouse moved, add velocity
                if is_painting and prev_mouse_pos is not None:
                    self._add_velocity_from_mouse(prev_mouse_pos, mouse_pos)

                # Update previous mouse position
                prev_mouse_pos = mouse_pos if is_painting else None

                for event in pygame.event.get():  # allow exit
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.MOUSEBUTTONDOWN:  # for debugging
                        # Get mouse position (pixels)
                        mx, my = pygame.mouse.get_pos()

                        # Convert to grid coordinates
                        j = mx // self.display_size
                        i = my // self.display_size

                        cell_edges = self.grid[i][j].get_edges_index()
                        print(f"Clicked cell: ({i}, {j})")
                        self.velocities_y[cell_edges[2][0]][cell_edges[2][1]] += 0.2
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
                self.clock.tick(30)  # framerate

            step += 1

    def _add_velocity_from_mouse(self, prev_pos, curr_pos):
        """
        Add velocity to the grid based on mouse movement.
        The velocity direction and magnitude are determined by mouse movement.
        """
        # Convert pixel positions to grid coordinates
        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos

        # Calculate mouse velocity in pixels
        dx_pixels = curr_x - prev_x
        dy_pixels = curr_y - prev_y

        # Convert to grid units (cells per frame)
        dx_grid = dx_pixels / self.display_size
        dy_grid = dy_pixels / self.display_size

        # Scale velocity (adjust multiplier to control strength)
        velocity_scale = 5.0
        u_add = dx_grid * velocity_scale
        v_add = dy_grid * velocity_scale

        # Get current cell position
        j = curr_x // self.display_size
        i = curr_y // self.display_size

        # Check bounds
        if i < 1 or i >= self.height - 1 or j < 1 or j >= self.width - 1:
            return

        # Apply velocity to a brush area (3x3 cells for better visibility)
        brush_radius = 1
        for di in range(-brush_radius, brush_radius + 1):
            for dj in range(-brush_radius, brush_radius + 1):
                ni = i + di
                nj = j + dj

                # Check bounds
                if ni < 1 or ni >= self.height - 1 or nj < 1 or nj >= self.width - 1:
                    continue

                # Skip solid cells
                if self.grid[ni][nj].is_solid:
                    continue

                # Get cell edges
                cell_edges = self.grid[ni][nj].get_edges_index()

                # Add horizontal velocity to left and right edges
                self.velocities_x[cell_edges[0][0]][cell_edges[0][1]] += u_add
                self.velocities_x[cell_edges[1][0]][cell_edges[1][1]] += u_add

                # Add vertical velocity to top and bottom edges
                self.velocities_y[cell_edges[2][0]][cell_edges[2][1]] += v_add
                self.velocities_y[cell_edges[3][0]][cell_edges[3][1]] += v_add

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
        self._vel_advect()  # Self-advection: velocity field advects itself
        self._vel_project()  # Remove divergence (make incompressible)

    def _dens_diffuse(self):
        a = (
            self.dt * self.diffusion / (self.cell_size * self.cell_size)
        )  # controls rate of approach/equalisation -> faster if timesteps larger, diffusion higher, cell size smaller
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
        # N = (
        #     max(self.height, self.width) - 2
        # )  # Internal grid size (excluding boundaries)
        dt0 = (
            self.dt / self.cell_size
        )  # used to determine how many cells we need to go back -> since speed is in m/s, so we need to go more cells back if cells are smaller

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
                    t0 * self.grid[i0][j0].density + t1 * self.grid[i1][j0].density
                ) + s1 * (
                    t0 * self.grid[i0][j1].density + t1 * self.grid[i1][j1].density
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
            self.grid[i][self.width - 1].next_density = self.grid[i][
                self.width - 2
            ].next_density

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
            self.grid[1][self.width - 1].next_density
            + self.grid[0][self.width - 2].next_density
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
        dt0 = (
            self.dt / self.cell_size
        )  # used to determine how many cells we need to go back

        # Advect horizontal velocities (velocities_x) - sampled at left edge centers
        for i in range(self.height):
            for j in range(self.width + 1):
                # Skip if adjacent to solid cells
                if j > 0 and self.grid[i][j - 1].is_solid:
                    self.next_velocities_x[i][j] = self.velocities_x[i][j]
                    continue
                if j < self.width and self.grid[i][j].is_solid:
                    self.next_velocities_x[i][j] = self.velocities_x[i][j]
                    continue

                # Sample at left edge center position (x=j, y=i+0.5 in grid coordinates)
                x = j
                y = i + 0.5
                u, v = self._get_velocity_at(x, y)

                # Trace particle backward in time
                x -= dt0 * u
                y -= dt0 * v

                # Sample velocity at the previous position
                u_prev, _ = self._get_velocity_at(x, y)
                self.next_velocities_x[i][j] = u_prev

        # Advect vertical velocities (velocities_y) - sampled at bottom edge centers
        for i in range(self.height + 1):
            for j in range(self.width):
                # Skip if adjacent to solid cells
                if i > 0 and self.grid[i - 1][j].is_solid:
                    self.next_velocities_y[i][j] = self.velocities_y[i][j]
                    continue
                if i < self.height and self.grid[i][j].is_solid:
                    self.next_velocities_y[i][j] = self.velocities_y[i][j]
                    continue

                # Sample at bottom edge center position (x=j+0.5, y=i in grid coordinates)
                x = j + 0.5
                y = i
                u, v = self._get_velocity_at(x, y)

                # Trace particle backward in time
                x -= dt0 * u
                y -= dt0 * v

                # Sample velocity at the previous position
                _, v_prev = self._get_velocity_at(x, y)
                self.next_velocities_y[i][j] = v_prev

        # Update velocities from temporary arrays
        self._update_velocities()

    def _update_velocities(self):
        """Copy next_velocities to current velocities"""
        for i in range(self.height):
            for j in range(self.width + 1):
                self.velocities_x[i][j] = self.next_velocities_x[i][j]

        for i in range(self.height + 1):
            for j in range(self.width):
                self.velocities_y[i][j] = self.next_velocities_y[i][j]

    def _get_divergence(self, i, j):
        div = 0.0
        vel_edges = self.grid[i][j].get_edges_index()
        div -= self.velocities_x[vel_edges[0][0]][vel_edges[0][1]]  # left edge - in
        div += self.velocities_x[vel_edges[1][0]][vel_edges[1][1]]  # right edge - out
        div -= self.velocities_y[vel_edges[2][0]][vel_edges[2][1]]  # up edge - in
        div += self.velocities_y[vel_edges[3][0]][vel_edges[3][1]]  # down edge - out
        return div / self.cell_size

    def _vel_project(self):
        self._pressure = [[0.0 for _ in range(self.width)] for _ in range(self.height)]

        def get_pressures(i, j):
            if self.grid[i][j].is_solid:
                return []
            neighbors = []
            if j > 0 and not self.grid[i][j - 1].is_solid:  # left
                neighbors.append(self._pressure[i][j - 1])
            if j < self.width - 1 and not self.grid[i][j + 1].is_solid:  # right
                neighbors.append(self._pressure[i][j + 1])
            if i > 0 and not self.grid[i - 1][j].is_solid:  # up
                neighbors.append(self._pressure[i - 1][j])
            if i < self.height - 1 and not self.grid[i + 1][j].is_solid:  # down
                neighbors.append(self._pressure[i + 1][j])
            return neighbors

        for _ in range(30):  # Gauss Seidel iterations
            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i][j].is_solid:
                        self._pressure[i][j] = 0
                    else:
                        pressures = get_pressures(i, j)
                        self._pressure[i][j] = (
                            (
                                (
                                    sum(pressures)
                                    - (
                                        self._get_divergence(i, j)
                                        * self.cell_size
                                        * self.cell_size
                                        / self.dt
                                    )
                                )
                                / len(pressures)
                            )
                            if pressures
                            else 0.0
                        )

        for i in range(self.height):
            for j in range(1, self.width):
                # Horizontal velocities
                if self.grid[i][j - 1].is_solid or self.grid[i][j].is_solid:
                    continue

                p_right = self._pressure[i][j]
                p_left = self._pressure[i][j - 1]
                self.velocities_x[i][j] -= (p_right - p_left) * self.dt / self.cell_size

        for i in range(1, self.height):
            for j in range(self.width):
                # Vertical velocities
                if self.grid[i - 1][j].is_solid or self.grid[i][j].is_solid:
                    continue

                p_down = self._pressure[i][j]
                p_up = self._pressure[i - 1][j]
                self.velocities_y[i][j] -= (p_down - p_up) * self.dt / self.cell_size

    def _draw_grid(self):
        self.screen.fill((0, 0, 0))  # clear background

        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                # Draw cell color (density)
                if cell.is_solid:
                    color = (0, 0, 0)
                else:
                    if self.show_cell_property == "density":
                        GRAY_VALUE = 30
                        val = max(
                            GRAY_VALUE,
                            GRAY_VALUE
                            + min(
                                255 - GRAY_VALUE, int(cell.density * (255 - GRAY_VALUE))
                            ),
                        )
                        color = (val, val, val)
                    elif self.show_cell_property == "divergence":
                        div = self._get_divergence(y, x)
                        color = (
                            max(0, min(255, int(div))),
                            0,
                            max(0, min(255, int(-div))),
                        )
                    elif self.show_cell_property == "pressure":
                        # print(self._pressure)
                        pressure = self._pressure[y][x]
                        color = (
                            max(0, min(255, int(200 * pressure))),
                            0,
                            max(0, min(255, int(200 * -pressure))),
                        )
                    elif self.show_cell_property == "advection":
                        # Show neutral background to highlight velocity arrows
                        GRAY_VALUE = 30
                        color = (GRAY_VALUE, GRAY_VALUE, GRAY_VALUE)
                rect = pygame.Rect(
                    x * self.display_size,
                    y * self.display_size,
                    self.display_size,
                    self.display_size,
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
                        scale = self.display_size * 0.4

                        # Arrow end points
                        start_x = (j + 0.5) * self.display_size
                        start_y = (i + 0.5) * self.display_size
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
                            points=[
                                (end_x, end_y),
                                (left_x, left_y),
                                (right_x, right_y),
                            ],
                            closed=True,
                        )

            elif not self.show_cell_centered_velocity:
                # Velocity for each face
                # Horizontal Velocity
                for i in range(self.height):
                    for j in range(self.width + 1):
                        # Get and clamp velocity
                        mag_dir = self.velocities_x[i][j]
                        mag_dir = (
                            min(1.0, mag_dir) if mag_dir > 0 else max(-1.0, mag_dir)
                        )

                        # Map color based on magnitude
                        r = int(255 * abs(mag_dir))
                        g = 0
                        b = int(255 * (1 - abs(mag_dir)))
                        color = (r, g, b)
                        # Scale arrow as per grid size for visibility
                        scale = self.display_size * 0.4

                        # Arrow end points
                        start_x = j * self.display_size
                        end_x = start_x + scale * mag_dir
                        y = (i + 0.5) * self.display_size

                        pygame.draw.aaline(
                            self.screen, color, (start_x, y), (end_x, y), 2
                        )

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
                            closed=True,
                        )

                # Vertical Velocity
                for i in range(self.height + 1):
                    for j in range(self.width):
                        # Get and clamp velocity
                        mag_dir = self.velocities_y[i][j]
                        mag_dir = (
                            min(1.0, mag_dir) if mag_dir > 0 else max(-1.0, mag_dir)
                        )

                        # Map color based on magnitude
                        r = int(255 * abs(mag_dir))
                        g = 0
                        b = int(255 * (1 - abs(mag_dir)))
                        color = (r, g, b)
                        # Scale arrow as per grid size for visibility
                        scale = self.display_size * 0.4

                        # Arrow end points
                        x = (j + 0.5) * self.display_size
                        start_y = i * self.display_size
                        end_y = start_y + scale * mag_dir

                        pygame.draw.aaline(
                            self.screen, color, (x, start_y), (x, end_y), 2
                        )

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
                            closed=True,
                        )

        # cursor indicator when in painting mode
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # circle to show brush area
            brush_radius_pixels = self.display_size * 1.5
            pygame.draw.circle(
                self.screen,
                (0, 255, 0),
                (mouse_x, mouse_y),
                int(brush_radius_pixels),
                2,
            )
            # crosshair
            pygame.draw.line(
                self.screen,
                (0, 255, 0),
                (mouse_x - 10, mouse_y),
                (mouse_x + 10, mouse_y),
                2,
            )
            pygame.draw.line(
                self.screen,
                (0, 255, 0),
                (mouse_x, mouse_y - 10),
                (mouse_x, mouse_y + 10),
                2,
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
        # Clamp to valid range
        px_clamped = max(0.0, min(px, self.width))
        py_clamped = max(0.0, min(py, self.height))

        # Check for solid at clamped position
        grid_i = int(py_clamped)
        grid_j = int(px_clamped)
        if grid_i >= self.height:
            grid_i = self.height - 1
        if grid_j >= self.width:
            grid_j = self.width - 1

        if self.grid[grid_i][grid_j].is_solid:
            return 0.0, 0.0
        u = self._sample_bilinear(
            self.velocities_x, px_clamped, py_clamped - 0.5, self.height, self.width + 1
        )
        v = self._sample_bilinear(
            self.velocities_y, px_clamped - 0.5, py_clamped, self.height + 1, self.width
        )
        return u, v


def f(x, y):
    return ((x - 10) ** 2 + (y - 25) ** 2) ** 1 / 2 - 10


if __name__ == "__main__":
    grid = FluidGrid(
        height=20,
        width=100,
        cell_size=0.01,
        diffusion=0.001,
        viscosity=0.01,
        dt=0.01,
        sources=[(10, 10, 300)],
        sdf=f,
        field_type="wind tunnel",
        visualise=True,
        show_cell_property="density",
        show_velocity=True,
        show_cell_centered_velocity=False,
    )

    grid.simulate()
