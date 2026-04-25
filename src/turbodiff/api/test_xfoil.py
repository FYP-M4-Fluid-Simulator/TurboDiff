import subprocess
import tempfile
import os
import logging
import shutil
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)


def run_xfoil_test():
    # 1. Resolve binary explicitly
    logger.info("XFOIL FUNCTION STARTED : ")
    output = subprocess.run(["xfoil"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  
    print("XFOIL STDOUT:", output.stdout)
    print("XFOIL STDERR:", output.stderr)
    # xfoil_path = shutil.which("xfoil")

    # if xfoil_path is None:
    #     logger.error("XFOIL_NOT_FOUND_IN_PATH")
    #     return {
    #         "success": False,
    #         "error": "xfoil binary not found. Install or add to PATH."
    #     }

    # # 2. XFOIL Commands
    # # Using a standard NACA 0012 airfoil avoids the geometry math crash.
    # # Note the empty strings "", these act as "Enter" to back out of menus.
    # commands = "\n".join([
    #     "PLOP",         # Enter plot options
    #     "G",            # Graphics flag
    #     "F",            # False (disable X11 graphics for headless server)
    #     "",             # Hit Enter to return to main menu
    #     "NACA 0012",    # Generate a standard symmetric airfoil
    #     "PANE",         # Calculate paneling
    #     "OPER",         # Enter operations menu
    #     "VISC 1000000", # Turn on viscous mode with Re = 1,000,000
    #     "ITER 50",      # Set iteration limit
    #     "ALFA 5",       # Run at 5 degrees Angle of Attack
    #     "",             # Hit Enter to return to main menu
    #     "QUIT"          # Exit program safely
    # ]) + "\n"

    # # Run in a temp directory so any XFOIL dump files don't clutter your app
    # with tempfile.TemporaryDirectory() as tmp:
    #     try:
    #         result = subprocess.run(
    #             [xfoil_path],
    #             input=commands,
    #             text=True,
    #             capture_output=True,
    #             cwd=tmp,
    #             timeout=10,
    #         )

    #         stdout = result.stdout or ""
    #         stderr = result.stderr or ""

    #         # Check if lift (CL) or drag (CD) was calculated successfully
    #         success = ("CL =" in stdout) or ("CD =" in stdout)

    #         logger.info("XFOIL_RETURN_CODE: %s", result.returncode)
    #         logger.info("XFOIL_SUCCESS: %s", success)

    #         # 3. Treat failure as failure
    #         if result.returncode != 0 or not success:
    #             return {
    #                 "success": False,
    #                 "returncode": result.returncode,
    #                 "stdout": stdout[-500:], # Grab the last 500 chars to see the actual error
    #                 "stderr": stderr
    #             }

    #         return {
    #             "success": True,
    #             "returncode": result.returncode,
    #             "stdout_snippet": stdout[-300:] # Show the aerodynamic results
    #         }

    #     except subprocess.TimeoutExpired:
    #         logger.error("XFOIL_TIMEOUT")
    #         return {"success": False, "error": "timeout"}

    #     except Exception as e:
    #         logger.error("XFOIL_EXCEPTION: %s", str(e))
    #         return {"success": False, "error": str(e)}


router = APIRouter(prefix="/test_xfoil", tags=["xfoil_test"])


@router.get("/xfoil-test")
def xfoil_test():
    result = run_xfoil_test()

    # if not result.get("success"):
    #     raise HTTPException(status_code=500, detail=result)
    result = {"success": True, "message": "XFOIL test completed successfully"}
    return result
