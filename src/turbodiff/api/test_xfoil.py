from __future__ import annotations
import os
import subprocess
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Mock or Import your CST generator
try:
    from turbodiff.core.airfoil import generate_cst_coords
except ImportError:
    def generate_cst_coords(wu, wl, num_points):
        x = jnp.linspace(0, 1, num_points)
        return x, jnp.zeros_like(x), jnp.zeros_like(x)

# --- Logic Classes ---

@dataclass(frozen=True)
class XFoilResult:
    cl: float
    cd: float
    cm: float
    aoa: float
    re: float

class XFoilRunner:
    def __init__(self):
        self.binary = self._find_binary()
        if not self.binary:
            raise RuntimeError("XFoil binary not found. Is it installed?")

    def _find_binary(self) -> str:
        return shutil.which("xfoil") or os.environ.get("XFOIL_PATH", "")

    def run_sim(self, dat_path: Path, re: float, aoa: float) -> Optional[XFoilResult]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            td = Path(tmp_dir)
            local_dat = td / "airfoil.dat"
            shutil.copy(dat_path, local_dat)
            polar = td / "polar.txt"

            cmds = [
                "PLOP", "G F", "", 
                f"LOAD {local_dat.name}", "PANE", "OPER",
                f"VISC {re}", "ITER 300", "PACC",
                polar.name, "", f"ALFA {aoa}", "QUIT"
            ]

            subprocess.run(
                [self.binary], input="\n".join(cmds) + "\n",
                text=True, capture_output=True, cwd=tmp_dir, timeout=20
            )

            if not polar.exists():
                return None

            with open(polar, "r") as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 5:
                        try:
                            if abs(float(p[0]) - aoa) < 0.01:
                                return XFoilResult(float(p[1]), float(p[2]), float(p[4]), aoa, re)
                        except ValueError: continue
            return None

# --- Router Setup ---

router = APIRouter(prefix="/xfoil", tags=["Aerodynamics"])
runner = XFoilRunner()

class SimulationInput(BaseModel):
    weights_upper: List[float] = Field(..., min_items=6)
    weights_lower: List[float] = Field(..., min_items=6)
    reynolds: float = Field(100000.0, gt=0)
    aoa: float = Field(0.0)

class SimulationOutput(BaseModel):
    cl: Optional[float]
    cd: Optional[float]
    ld_ratio: Optional[float]
    converged: bool

@router.post("/xfoil_run", response_model=SimulationOutput)
async def execute_xfoil(data: SimulationInput):
    # Idempotent file creation
    tmp_file = Path(f"sim_{uuid.uuid4()}.dat")
    try:
        # Write Selig Format
        wu, wl = jnp.array(data.weights_upper), jnp.array(data.weights_lower)
        x, yu, yl = generate_cst_coords(wu, wl, 200)
        
        with open(tmp_file, "w") as f:
            f.write("CST_AIRFOIL\n")
            for xi, yi in zip(reversed(np.array(x)), reversed(np.array(yu))):
                f.write(f" {xi:.7f} {yi:.7f}\n")
            for xi, yi in zip(np.array(x)[1:], np.array(yl)[1:]):
                f.write(f" {xi:.7f} {yi:.7f}\n")

        res = runner.run_sim(tmp_file, data.reynolds, data.aoa)
        
        if not res:
            return SimulationOutput(cl=None, cd=None, ld_ratio=None, converged=False)

        return SimulationOutput(
            cl=res.cl, cd=res.cd, 
            ld_ratio=res.cl/res.cd if res.cd != 0 else 0, 
            converged=True
        )
    finally:
        if tmp_file.exists():
            tmp_file.unlink()

@router.get("/xfoil_status")
async def check_status():
    return {"binary_found": bool(runner.binary), "path": runner.binary}