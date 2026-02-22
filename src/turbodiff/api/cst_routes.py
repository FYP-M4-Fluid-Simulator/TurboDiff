"""all CST related endpoints"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from turbodiff.utils.dat_to_cst import CST_fitting
from turbodiff.db.storage import get_storage_repository
import tempfile
import os
import shutil

router = APIRouter()


@router.get("/cst")
def list_cst_for_user(user_id: str):
    repo = get_storage_repository()
    csts = repo.list_cst_for_user(user_id)
    print(f"Listing CSTs for user {user_id}: Found {csts}")
    return {
        "items": [
            {
                "id": cst.id,
                "name" : f"Airfoil {i}",
                "weights_upper": cst.weights_upper,
                "weights_lower": cst.weights_lower,
                "chord_length": cst.chord_length,
                "cst_created_at": cst.cst_created_at.isoformat(),
                "cl": cst.cl,
                "cd": cst.cd,
                "lift": cst.lift,
                "drag": cst.drag,
                "angle_of_attack": cst.angle_of_attack,
                "created_by_user_id": cst.created_by_user_id,
                "is_optimized": cst.is_optimized,
                "airfoil_created_at": cst.airfoil_created_at.isoformat(),
            }
            for i, cst in enumerate(csts)
        ]
    }


@router.post("/get_cst_values")
async def get_cst_values(
    file: UploadFile = File(...),
    bernenstein_order: int = 8,
    leading_edge_radius: float = 0.015867,
):
    temp_file_path = None
    output_file_path = None

    try:
        # 1. Check file extension
        if not file.filename.endswith(".dat"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only .dat files are supported.",
            )

        print(
            f"Received file: {file.filename} with Bernstein order: {bernenstein_order}"
        )

        # 2. Create a specific temp file (Delete=False is mandatory for Windows)
        # We use 'wb' (Write Binary) to prevent newline corruption
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".dat", delete=False
        ) as temp_file:
            # Copy the upload directly to disk without decoding (fast & safe)
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
            # The file is automatically closed here when we exit the 'with' block

        print(f"Saved temp file to: {temp_file_path}")

        # 3. Create output path
        output_file_path = temp_file_path.replace(".dat", "_fitted.dat")

        # 4. Call CST_fitting
        # Now that the temp file is closed, your CST script can open it safely

        try:
            output = CST_fitting(
                temp_file_path, output_file_path, bernenstein_order, leading_edge_radius
            )
            return {
                "upperCoefficients": output["upperCoefficients"],
                "lowerCoefficients": output["lowerCoefficients"],
                "accuracy": output["accuracy"],
                "message": output["message"],
            }
        except IndexError:
            # Catch the specific "Index out of range" error
            raise HTTPException(
                status_code=400,
                detail="Parsing Error: The .dat file format is invalid. Ensure there are no empty lines or text headers (like 'NACA0012') at the top.",
            )
        except Exception as e:
            print(f"CST Fitting Failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Fitting algorithm failed: {str(e)}"
            )

    finally:
        # 5. Cleanup (Always runs, even if error occurs)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass  # Ignore cleanup errors

        if output_file_path and os.path.exists(output_file_path):
            try:
                os.unlink(output_file_path)
            except Exception:
                pass
