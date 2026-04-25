import logging
import re
import shutil
import subprocess
import sys

from fastapi import APIRouter


router = APIRouter(prefix="/xfoil", tags=["xfoil"])


# Setup logging to see what happens inside the API
logger = logging.getLogger("XFoilAPI")
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@router.get("/xfoil_cl_cd")
def get_naca_cl_cd(naca_code: str, aoa: float, reynolds: float = 1000000):
    """
    Runs XFOIL for a given NACA airfoil and Angle of Attack (AoA).

    Parameters:
        naca_code (str): 4 or 5 digit NACA code (e.g., '0012')
        aoa (float): Angle of Attack in degrees
        reynolds (float): Reynolds number for viscous calculation

    Returns:
        tuple: (Cl, Cd) as floats, or (None, None) if it fails to converge.
    """
    # 1. Define the exact sequence of commands you would type into XFOIL
    logger.info(
        "Received XFOIL request: naca=%s aoa=%s reynolds=%s",
        naca_code,
        aoa,
        reynolds,
    )
    print(
        f"[XFoilAPI] Running XFOIL for naca={naca_code}, aoa={aoa}, reynolds={reynolds}",
        flush=True,
    )

    xfoil_commands = f"""
naca {naca_code}
oper
iter 150
visc {reynolds}
alfa {aoa}
quit
"""

    try:
        # 2. Run XFOIL as a subprocess
        # Note: Ensure the command 'xfoil' is in your container's system PATH
        process = subprocess.Popen(
            ["xfoil"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Pass the commands and capture the output
        stdout, stderr = process.communicate(input=xfoil_commands)
        logger.info("XFOIL process finished with return code: %s", process.returncode)

        if stderr:
            logger.warning("XFOIL stderr output: %s", stderr.strip())
            print(f"[XFoilAPI] XFOIL stderr: {stderr.strip()}", flush=True)

        # 3. Parse the output text using Regular Expressions
        # We are looking for the exact lines where XFOIL prints CL = ... and CD = ...
        cl_match = re.search(r"CL\s*=\s*([+-]?\d*\.\d+)", stdout)
        cd_match = re.search(r"CD\s*=\s*([+-]?\d*\.\d+)", stdout)

        if cl_match and cd_match:
            cl = float(cl_match.group(1))
            cd = float(cd_match.group(1))
            logger.info("XFOIL converged: Cl=%s Cd=%s", cl, cd)
            print(f"[XFoilAPI] XFOIL converged with Cl={cl}, Cd={cd}", flush=True)
            return cl, cd
        else:
            logger.warning(
                "Failed to parse Cl/Cd. The solution might not have converged for AoA=%s",
                aoa,
            )
            print(
                f"[XFoilAPI] Failed to find Cl/Cd in output for AoA={aoa}.",
                flush=True,
            )
            return None, None

    except FileNotFoundError:
        logger.exception(
            "'xfoil' command not found. Ensure it is installed and in PATH."
        )
        print("[XFoilAPI] Error: 'xfoil' command not found.", flush=True)
        return None, None
    except Exception:
        logger.exception("Unexpected error while running XFOIL")
        print("[XFoilAPI] Unexpected error while running XFOIL.", flush=True)
        return None, None


@router.get("/xfoil_status")
async def check_status():
    binary_path = shutil.which("xfoil")
    logger.info(
        "XFOIL status checked. binary_found=%s path=%s", bool(binary_path), binary_path
    )
    print(
        f"[XFoilAPI] Status check: binary_found={bool(binary_path)}, path={binary_path}",
        flush=True,
    )
    return {"binary_found": bool(binary_path), "path": binary_path}
