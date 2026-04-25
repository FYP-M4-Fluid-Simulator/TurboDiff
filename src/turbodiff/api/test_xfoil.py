from __future__ import annotations
import os
import subprocess
import shutil
import tempfile
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Setup logging to see what happens inside the API
logger = logging.getLogger("XFoilAPI")

# --- Import/Mock CST ---
try:
    from turbodiff.core.airfoil import generate_cst_coords
except ImportError:
    def generate_cst_coords(wu, wl, num_points=200):
        x = jnp.linspace(0, 1, num_points)
        yu = 0.1 * jnp.sin(np.pi * x) 
        yl = -0.1 * jnp.sin(np.pi * x)
        return x, yu, yl

# --- Logic from your working isolated file ---

@dataclass(frozen=True)
class XFoilResult:
    cl: float
    cd: float
    cm: float
    aoa: float
    re: float

class XFoilRunner:
    def __init__(self):
        self.binary = self._resolve_binary()
        if not self.binary:
            raise RuntimeError("XFoil binary not found.")

    def _resolve_binary(self) -> str:
        search = [shutil.which("xfoil"), os.environ.get("XFOIL_PATH", "")]
        return next((p for p in search if p and os.access(p, os.X_OK)), "")

    def run_sim(self, dat_path: Path, re: float, aoa: float) -> Optional[XFoilResult]:
        # Using a TemporaryDirectory ensures NO file collisions during API calls
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            local_dat = td / "airfoil.dat"
            shutil.copy(dat_path, local_dat)
            polar_file = td / "polar.txt"

            # EXACT same commands from the isolated script
            commands = [
                "PLOP", "G F", "", 
                f"LOAD {local_dat.name}",
                "PANE",
                "OPER",
                f"VISC {re}",
                "ITER 300",
                "PACC",
                polar_file.name, "", 
                f"ALFA {aoa}",
                "QUIT"
            ]

            try:
                subprocess.run(
                    [self.binary],
                    input="\n".join(commands) + "\n",
                    text=True, capture_output=True, cwd=tmp_dir, timeout=30
                )
                return self._parse_polar(polar_file, aoa, re)
            except Exception as e:
                logger.error(f"XFoil Error: {e}")
                return None

    def _parse_polar(self, path: Path, target_aoa: float, re: float) -> Optional[XFoilResult]:
        if not path.exists(): return None
        with open(path, "r") as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    try:
                        if abs(float(p[0]) - target_aoa) < 0.01:
                            return XFoilResult(float(p[1]), float(p[2]), float(p[4]), target_aoa, re)
                    except ValueError: continue
        return None

# --- FastAPI Router ---

router = APIRouter(prefix="/xfoil", tags=["Simulation"])
runner = XFoilRunner()

class SimInput(BaseModel):
    weights_upper: List[float] = Field(..., min_items=6)
    weights_lower: List[float] = Field(..., min_items=6)
    reynolds: float = Field(10000.0) # Defaulting to your working Re
    aoa: float = Field(0.0)

class SimOutput(BaseModel):
    cl: Optional[float]
    cd: Optional[float]
    converged: bool

@router.post("/run", response_model=SimOutput)
async def run_xfoil_api(data: SimInput):
    # Use a unique name in the system's temp folder
    temp_dat = Path(tempfile.gettempdir()) / f"api_sim_{uuid.uuid4().hex}.dat"
    
    try:
        # 1. Generate Coords
        wu, wl = jnp.array(data.weights_upper), jnp.array(data.weights_lower)
        x, yu, yl = generate_cst_coords(wu, wl, 200)
        
        # 2. Write File
        with open(temp_dat, "w") as f:
            f.write("API_AIRFOIL\n")
            for xi, yi in zip(reversed(np.array(x)), reversed(np.array(yu))):
                f.write(f" {xi:.7f} {yi:.7f}\n")
            for xi, yi in zip(np.array(x)[1:], np.array(yl)[1:]):
                f.write(f" {xi:.7f} {yi:.7f}\n")

        # 3. Run
        res = runner.run_sim(temp_dat, data.reynolds, data.aoa)
        
        if not res:
            return SimOutput(cl=None, cd=None, converged=False)

        return SimOutput(cl=res.cl, cd=res.cd, converged=True)

    finally:
        if temp_dat.exists():
            temp_dat.unlink()


            
@router.get("/xfoil_status")
async def check_status():
    return {"binary_found": bool(runner.binary), "path": runner.binary}