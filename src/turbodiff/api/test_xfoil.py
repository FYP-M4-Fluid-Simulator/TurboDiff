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

    # 1. Hardened XFOIL commands
    # We add an inviscid initialization first to give the viscous solver a stable starting guess.
    commands = f"""
    PLOP
    G F

    NACA {naca}
    PANE
    OPER
    ALFA {alfa}       
    VISC {reynolds}
    ITER 200
    ALFA {alfa}

    QUIT
    """

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
            return {"success": False, "error": "XFOIL timed out (solution did not converge)."}

    # 2. Gracefully handle the Floating-Point Exception (-8)
    if result.returncode == -8:
        logger.warning(f"XFOIL math crashed (SIGFPE) for NACA {naca} at ALFA {alfa}")
        return {
            "success": False, 
            "error": "Aerodynamic math crashed. The angle of attack is likely too high (stall), causing the flow equations to fail."
        }
    elif result.returncode != 0:
        return {"success": False, "error": f"XFOIL failed with return code {result.returncode}"}

    # 3. Extract CL and CD
    cl_match = re.search(r"CL\s*=\s*([-\d.]+)", result.stdout, re.IGNORECASE)
    cd_match = re.search(r"CD\s*=\s*([-\d.]+)", result.stdout, re.IGNORECASE)

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
        return {"success": False, "error": "Solution did not converge. Could not calculate CL and CD."}

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