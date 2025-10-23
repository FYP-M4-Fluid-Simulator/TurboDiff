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
2. Install turbodiff in editable mode using `pip install -e .[dev]` from project root directory
3. Run `pre-commit install`, this will ensure that your code is reformatted according to the Black formatter on commit

# Testing

1. Add tests in test directory
2. Run tests using command `pytest`

# Documentation

1. Modify files in docs
2. Compile docs by going into `docs/` and running `make html` command

# Coding Conventions

1. PascalCase for classes
2. snake_case for variables and functions
3. imports at the top
