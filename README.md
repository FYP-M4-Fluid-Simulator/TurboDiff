# TurboDiff
A differentiable fluid simulation pipeline for Wind Turbine Shape Optimization


# Intended Approach
1. Simulate 2D smooth wind flow
2. Ensure differentiability of the above with parametrised 2D airfoils
3. Use Blade Element Momentum (BEM) methodology for connecting slice geometry to results for 3D wind turbines, incorporating BEM correctors
4. Leverage differentiability to optimize parameterised shape of each slice to maximise turbine efficiency while maintaining some design & physical constraints
5. Enhance wind flow simulation with low order RANS