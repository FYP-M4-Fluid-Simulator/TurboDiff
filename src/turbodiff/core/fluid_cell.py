class FluidCell:
    """
    Fluid cells store density and whether they are solid
    Density can vary between [0, 1]
    Solid cells don't have density and can't act as source. Will also have no velocity on their edges
    """

    def __init__(
        self, cell_index: tuple[int, int], density: float, is_solid: bool = False
    ):
        self.cell_index = cell_index
        self.is_solid = is_solid

        # need prev value for calculation
        if not self.is_solid:
            self.density = density
            self.next_density = density
        else:
            self.density = 0
            self.next_density = 0

        self.source = 0  # may be set later

    def set_source(self, s):
        if self.is_solid:
            raise TypeError("Solid cell can not be a source")
        self.source = s

    def add_source(self, dt):
        self.next_density += dt * self.source

    def update_cell(self):
        self.density = self.next_density

    def get_edges_index(self):
        """Returns veloocity edge indexes for: (left vel x, right vel x, up vel y, down vel y)"""
        i, j = self.cell_index
        return ((i, j), (i, j + 1), (i, j), (i + 1, j))
