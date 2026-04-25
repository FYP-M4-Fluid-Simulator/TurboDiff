import subprocess
import tempfile
import shutil
import re
import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/xfoil", tags=["xfoil"])

def get_airfoil_data(naca="0012", alfa=5.0, reynolds=1000000):
    """Runs XFOIL for a given NACA profile and Angle of Attack, returning CL and CD."""
    xfoil_path = shutil.which("xfoil")
    
    if not xfoil_path:
        raise RuntimeError("XFOIL binary not found in system PATH.")

    # 1. Format the XFOIL commands
    commands = f"""
    PLOP
    G F

    NACA {naca}
    PANE
    OPER
    VISC {reynolds}
    ITER 100
    ALFA {alfa}

    QUIT
    """

    # 2. Execute in a temporary directory to avoid file clutter
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = subprocess.run(
                [xfoil_path],
                input=commands,
                text=True,
                capture_output=True,
                cwd=tmp,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("XFOIL timed out. The solution likely did not converge.")

    # 3. Handle hard crashes
    if result.returncode != 0:
        raise RuntimeError(f"XFOIL crashed with return code {result.returncode}")

    # 4. Use Regex to scrape the final CL and CD values from the stdout text
    # It looks for "CL = [number]" and extracts just the number.
    cl_match = re.search(r"CL\s*=\s*([-\d.]+)", result.stdout, re.IGNORECASE)
    cd_match = re.search(r"CD\s*=\s*([-\d.]+)", result.stdout, re.IGNORECASE)

    # 5. Validate and return
    if cl_match and cd_match:
        return {
            "success": True,
            "naca": naca,
            "angle_of_attack": alfa,
            "reynolds_number": reynolds,
            "cl": float(cl_match.group(1)),
            "cd": float(cd_match.group(1))
        }
    else:
        # If regex fails, it means XFOIL ran but the aerodynamics didn't converge
        logger.error("XFOIL Output Snippet: %s", result.stdout[-400:])
        raise RuntimeError("Aerodynamics did not converge. Could not calculate CL and CD.")


@router.get("/analyze")
def analyze_airfoil(naca: str = "0012", alfa: float = 5.0):
    """
    FastAPI Endpoint: 
    Example usage: /xfoil/analyze?naca=4412&alfa=3.5
    """
    try:
        data = get_airfoil_data(naca=naca, alfa=alfa)
        return data
    except Exception as e:
        logger.error("XFOIL_ERROR: %s", str(e))
        raise HTTPException(status_code=500, detail={"success": False, "error": str(e)})