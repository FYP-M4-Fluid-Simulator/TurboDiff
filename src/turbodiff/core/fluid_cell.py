class FluidCell:
    def __init__(self, density: float, velocity: float, is_solid: bool = False):
        # need prev value for calculation
        if not is_solid:
            self.prev_density = density
            self.density = density
            
            self.prev_velocity = 0
            self.velocity = velocity
        
    def set_density(self, density):
        self.density = density
            
    def set_velocity(self, velocity):
        self.velocity = velocity
    
    def update_cell(self):
        self.prev_density = self.density
        self.prev_velocity = self.velocity