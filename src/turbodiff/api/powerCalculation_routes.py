import math
from typing import Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# --- Constants ---
# Standard air density at sea level in kg/m^3
AIR_DENSITY = 1.225  

router = APIRouter(tags=["Wind Turbine Calculation"])

# --- Pydantic Models ---
class FullRotorInput(BaseModel):
    """
    Data model representing the inputs required from the frontend 
    to calculate power for the entire wind turbine.
    """
    v_wind: float = Field(..., gt=0, description="Incoming wind speed in m/s")
    alpha_deg: float = Field(..., description="Angle of attack in degrees")
    cl: float = Field(..., description="Lift coefficient (from simulator)")
    cd: float = Field(..., description="Drag coefficient (from simulator)")
    chord: float = Field(..., gt=0, description="Average chord length of the blade in meters")
    rpm: float = Field(..., ge=0, description="Rotational speed of the turbine in RPM")
    
    # New Geometric Inputs for the whole rotor
    rotor_radius: float = Field(..., gt=0, description="Total radius of the turbine from hub center to blade tip in meters")
    hub_radius: float = Field(1.0, ge=0, description="Radius of the central hub where no blade exists, in meters")
    num_blades: int = Field(3, gt=0, description="Number of blades on the turbine")
    num_elements: int = Field(20, gt=0, description="Number of segments to slice the blade into for calculation")

    class Config:
        json_schema_extra = {
            "example": {
                "v_wind": 12.0,
                "alpha_deg": 5.5,
                "cl": 0.85,
                "cd": 0.04,
                "chord": 1.5,
                "rpm": 15.0,
                "rotor_radius": 40.0,
                "hub_radius": 2.0,
                "num_blades": 3,
                "num_elements": 20
            }
        }

# --- Core Business Logic (Math) ---
def calculate_element_power(
    v_wind: float, rpm: float, radius: float, dr: float, chord: float, cl: float, cd: float
) -> float:
    """Calculates the mechanical power for a single blade slice."""
    if rpm == 0 or radius == 0:
        return 0.0

    omega = rpm * (math.pi / 30.0)
    
    # Calculate Apparent Wind Vectors for this specific radius
    phi_rad = math.atan(v_wind / (omega * radius))
    w_velocity = math.sqrt(v_wind**2 + (omega * radius)**2)
    
    # Calculate Forces
    dynamic_pressure = 0.5 * AIR_DENSITY * (w_velocity**2) * chord
    lift_prime = dynamic_pressure * cl
    drag_prime = dynamic_pressure * cd
    
    # Tangential force driving the rotor
    f_tangential = (lift_prime * math.sin(phi_rad)) - (drag_prime * math.cos(phi_rad))
    
    # Torque and Power
    torque_element = f_tangential * radius * dr
    power_element = torque_element * omega
    
    return power_element

def calculate_total_rotor_power(data: FullRotorInput) -> float:
    """
    Slices the blade into elements, calculates power for each, 
    and sums them up for all blades.
    """
    # Calculate the actual length of the aerodynamic blade
    blade_length = data.rotor_radius - data.hub_radius
    
    # Determine the length of each individual slice (dr)
    dr = blade_length / data.num_elements
    
    total_single_blade_power = 0.0
    
    # Loop through each slice of the blade
    for i in range(data.num_elements):
        # Find the center radius of the current slice
        # We use (i + 0.5) to get the midpoint of the element rather than the edge
        local_radius = data.hub_radius + (i + 0.5) * dr
        
        # In a fully advanced BEM model, you would look up a new CL and CD here 
        # because the inflow angle changes at every radius. For this script, 
        # we use the static inputs provided by the user.
        element_power = calculate_element_power(
            v_wind=data.v_wind,
            rpm=data.rpm,
            radius=local_radius,
            dr=dr,
            chord=data.chord,
            cl=data.cl,
            cd=data.cd
        )
        
        total_single_blade_power += element_power
        
    # Multiply by the number of blades to get the total turbine power
    total_turbine_power = total_single_blade_power * data.num_blades
    return total_turbine_power

# --- API Endpoints ---
@router.post("/power-output", response_model=Dict[str, float])
def power_output_calculation(inputs: FullRotorInput):
    """
    Endpoint to calculate the total power output of the wind turbine.
    """
    try:
        # Execute total rotor calculation
        total_power_watts = calculate_total_rotor_power(inputs)
        total_power_kw = total_power_watts / 1000.0
        
        return {
            "total_power_watts": round(total_power_watts, 2),
            "total_power_kilowatts": round(total_power_kw, 4)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during calculation: {str(e)}"
        )