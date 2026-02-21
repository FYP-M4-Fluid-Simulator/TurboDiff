# TurboDiff

A differentiable fluid simulation pipeline for Wind Turbine Shape Optimization

# Intended Approach

1. Simulate 2D smooth wind flow
2. Ensure differentiability of the above with parametrised 2D airfoils
3. Use Blade Element Momentum (BEM) methodology for connecting slice geometry to results for 3D wind turbines, incorporating BEM correctors
4. Leverage differentiability to optimize parameterised shape of each slice to maximise turbine efficiency while maintaining some design & physical constraints
5. Enhance wind flow simulation with low order RANS

# How to Setup for Development

1. Make a virtual environment for project using `python -m venv /path/to/venv`
2. Install turbodiff in editable mode using `pip install -e '.[dev]'` from project root directory
3. **(Linux with NVIDIA GPU only)** For GPU acceleration, install CUDA-enabled JAX:
   ```bash
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```
   Or use: `pip install -e '.[dev,cuda]'`
4. **(Windows)** JAX CPU-only is installed by default. GPU support is experimental.
5. Run `pre-commit install`, this will ensure that your code is reformatted according to the Black formatter on commit
6. Run `pre-commit run --all-files` to run black, Ruff, and prettier before committing

# Testing

1. Add tests in test directory
2. Run tests using command `pytest`

# Storage (Neon/Postgres)

TurboDiff now supports saving sessions and airfoils in Postgres (Neon compatible).

## Configure

1. Create a `.env` file at the repo root with:
   ```
   TURBODIFF_DATABASE_URL=postgresql://USER:PASSWORD@HOST/DB?sslmode=require
   ```
2. Install dependencies (includes `psycopg` and `python-dotenv`).

## Database migrations

Schema changes are applied via patch files. On startup, TurboDiff applies any
new patches that are not yet recorded in the database.

- Patch directory: `src/turbodiff/db/patches`
- Applied patch IDs are stored in `schema_migrations`
- Add new patches as `0002_*.sql`, `0003_*.sql`, ...

# Saved experiments API

Sessions and airfoils are saved automatically when creating sessions. Metrics
can be saved later.

## Simulate session (create)

```bash
curl -X POST http://localhost:8000/sessions \
   -H "Content-Type: application/json" \
   -d '{
      "user_id": "00000000-0000-0000-0000-000000000001",
      "fidelity": "medium",
      "sim_time": 2.0,
      "dt": 0.01,
      "cell_size": 0.01,
      "diffusion": 0.001,
      "viscosity": 0.0,
      "boundary_type": 1,
      "inflow_velocity": 2.0,
      "stream_fps": 30.0,
      "stream_every": 1,
      "angle_of_attack": 5.0,
      "cst_upper": [0.1, 0.2, 0.3],
      "cst_lower": [-0.1, -0.2, -0.3],
      "airfoil_offset_x": 0.2,
      "airfoil_offset_y": 0.32,
      "chord_length": 0.25,
      "num_cst_points": 100,
      "mask_sharpness": 50.0
   }'
```

## Simulate session (save metrics)

```bash
curl -X POST http://localhost:8000/sessions/{session_id}/save \
   -H "Content-Type: application/json" \
   -d '{
      "user_id": "00000000-0000-0000-0000-000000000001",
      "cl": 1.0,
      "cd": 0.1,
      "lift": 2.0,
      "drag": 0.2,
      "angle_of_attack": 4.0
   }'
```

## Optimize session (create)

```bash
curl -X POST http://localhost:8000/optimize/sessions \
   -H "Content-Type: application/json" \
   -d '{
      "user_id": "00000000-0000-0000-0000-000000000001",
      "fidelity": "low",
      "num_iterations": 30,
      "learning_rate": 0.005,
      "num_sim_steps": 80,
      "cst_upper": [0.18, 0.22, 0.20, 0.18, 0.15, 0.12],
      "cst_lower": [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03]
   }'
```

## Optimize session (save final airfoil)

```bash
curl -X POST http://localhost:8000/optimize/sessions/{session_id}/save \
   -H "Content-Type: application/json" \
   -d '{
      "user_id": "00000000-0000-0000-0000-000000000001",
      "cst_upper": [0.2, 0.22, 0.2],
      "cst_lower": [-0.1, -0.08, -0.06],
      "chord_length": 0.25,
      "angle_of_attack": 3.0,
      "cl": 1.2,
      "cd": 0.08,
      "lift": 2.5,
      "drag": 0.2
   }'
```

# Postgres integration test

Run the Postgres-backed repository test (uses `TURBODIFF_DATABASE_URL`):

```bash
TURBODIFF_DATABASE_URL="postgresql://USER:PASSWORD@HOST/DB?sslmode=require" \
pytest tests/test_storage_repository_postgres.py -m integration
```

Clean up the test data after running the integration test:

```bash
TURBODIFF_DATABASE_URL="postgresql://USER:PASSWORD@HOST/DB?sslmode=require" \
python scripts/cleanup_neon_test_data.py
```

# Documentation

1. Modify files in docs
2. Compile docs by going into `docs/` and running `make html` command

# Streaming API (FastAPI)

Run the WebSocket streaming server:

```bash
uvicorn turbodiff.api.app:app --reload
```

Create a session with a POST request (frontend defines conditions and shape):

```bash
curl -X POST http://localhost:8000/sessions \
   -H "Content-Type: application/json" \
   -d '{
      "fidelity": "medium",
      "sim_time": 2.0,
      "dt": 0.01,
      "cell_size": 0.01,
      "diffusion": 0.001,
      "viscosity": 0.0,
      "boundary_type": 1,
      "inflow_velocity": 2.0,
      "stream_fps": 30.0,
      "stream_every": 1,
      "angle_of_attack": 5.0,
      "cst_upper": [0.1, 0.2, 0.3],
      "cst_lower": [-0.1, -0.2, -0.3],
      "airfoil_offset_x": 0.2,
      "airfoil_offset_y": 0.32,
      "chord_length": 0.25,
      "num_cst_points": 100,
      "mask_sharpness": 50.0
   }'
```

Then connect to `ws://localhost:8000/ws/{session_id}` and the server will stream
cell-centered velocity and pressure fields as JSON arrays.

# Coding Conventions

1. PascalCase for classes
2. snake_case for variables and functions
3. imports at the top
