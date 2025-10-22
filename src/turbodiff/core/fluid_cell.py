class FluidCell:
    def __init__(
        self, density: float, velocity: tuple[float, float], is_solid: bool = False
    ):
        # need prev value for calculation
        if not is_solid:
            self.prev_density = density
            self.density = density

            self.prev_velocity = (0, 0)
            self.velocity = velocity

            self.source = 0  # may be set later

    def set_source(self, s):
        self.source = s

    def add_source(self, dt):
        self.density += dt * self.source

    def update_cell(self):
        self.prev_density = self.density
        self.prev_velocity = self.velocity
