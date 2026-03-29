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
    Data model representing the inputs required from the frontend.
    Notice 'rpm' is gone, replaced by 'tsr'.
    """
    v_wind: float = Field(..., gt=0, description="Incoming wind speed in m/s")
    alpha_deg: float = Field(..., description="Angle of attack in degrees")
    cl: float = Field(..., description="Lift coefficient (from simulator)")
    cd: float = Field(..., description="Drag coefficient (from simulator)")
    chord: float = Field(..., gt=0, description="Average chord length of the blade in meters")
    
    # Replaced RPM with TSR
    tsr: float = Field(7.0, gt=0, description="Target Tip Speed Ratio (lambda)")
    
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
                "tsr": 7.0,
                "rotor_radius": 40.0,
                "hub_radius": 2.0,
                "num_blades": 3,
                "num_elements": 20
            }
        }

# --- Core Business Logic (Math) ---
def calculate_target_rpm(tsr: float, v_wind: float, rotor_radius: float) -> float:
    """Calculates the required RPM to maintain the target Tip Speed Ratio."""
    return (tsr * v_wind * 60.0) / (2.0 * math.pi * rotor_radius)

def calculate_element_power(
    v_wind: float, rpm: float, radius: float, dr: float, chord: float, cl: float, cd: float
) -> float:
    """Calculates the mechanical power for a single blade slice."""
    if rpm == 0 or radius == 0:
        return 0.0

    omega = rpm * (math.pi / 30.0)
    
    phi_rad = math.atan(v_wind / (omega * radius))
    w_velocity = math.sqrt(v_wind**2 + (omega * radius)**2)
    
    dynamic_pressure = 0.5 * AIR_DENSITY * (w_velocity**2) * chord
    lift_prime = dynamic_pressure * cl
    drag_prime = dynamic_pressure * cd
    
    f_tangential = (lift_prime * math.sin(phi_rad)) - (drag_prime * math.cos(phi_rad))
    
    torque_element = f_tangential * radius * dr
    power_element = torque_element * omega
    
    return power_element

def calculate_total_rotor_power(data: FullRotorInput) -> tuple[float, float]:
    """
    Calculates the dynamic RPM, slices the blade, and sums the power.
    Returns a tuple of (total_power_watts, calculated_rpm).
    """
    # 1. Calculate the dynamic RPM based on the user's TSR
    operating_rpm = calculate_target_rpm(data.tsr, data.v_wind, data.rotor_radius)
    
    blade_length = data.rotor_radius - data.hub_radius
    dr = blade_length / data.num_elements
    total_single_blade_power = 0.0
    
    for i in range(data.num_elements):
        local_radius = data.hub_radius + (i + 0.5) * dr
        
        # Pass the newly calculated operating_rpm down to the element
        element_power = calculate_element_power(
            v_wind=data.v_wind,
            rpm=operating_rpm,
            radius=local_radius,
            dr=dr,
            chord=data.chord,
            cl=data.cl,
            cd=data.cd
        )
        total_single_blade_power += element_power
        
    total_turbine_power = total_single_blade_power * data.num_blades
    
    # Return both the power and the RPM we calculated
    return total_turbine_power, operating_rpm

# --- API Endpoints ---
@router.post("/power-output", response_model=Dict[str, float])
def power_output_calculation(inputs: FullRotorInput):
    """
    Endpoint to calculate the total power output and operational RPM of the wind turbine.
    """
    try:
        # Execute calculation, unpacking the tuple returned by the logic function
        total_power_watts, operating_rpm = calculate_total_rotor_power(inputs)
        
        total_power_kw = total_power_watts / 1000.0
        print("Calculated Power (Watts):", total_power_watts)
        print("Calculated Power (Kilowatts):", total_power_kw)
        print("Calculated RPM:", operating_rpm)
        # Return an enriched JSON payload so the frontend can display the RPM
        return {
            "total_power_watts": round(total_power_watts, 2),
            "total_power_kilowatts": round(total_power_kw, 4),
            "operating_rpm": round(operating_rpm, 2)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during calculation: {str(e)}"
        )