"""Validation tests for base CFD implementation (without SA-RANS)."""

from turbodiff.core.benchmark_validations import (
    run_blasius_boundary_layer_benchmark,
    run_taylor_green_vortex_benchmark,
)


def test_taylor_green_vortex_validation() -> None:
    """Taylor-Green should preserve flow structure and decay energy plausibly."""
    metrics = run_taylor_green_vortex_benchmark(
        n=36,
        dt=0.002,
        viscosity=0.01,
        u0=0.25,
        steps=100,
        pressure_iters=36,
        diffusion_iters=14,
        trim=2,
    )

    assert metrics.energy_ratio_rel_error <= 0.55
    assert metrics.corr_u >= 0.52
    assert metrics.corr_v >= 0.52
    assert metrics.div_l2 <= 0.22
    assert metrics.div_max <= 1.05


def test_blasius_boundary_layer_validation() -> None:
    """Flat-plate profile should be close to Blasius at a downstream station."""
    metrics = run_blasius_boundary_layer_benchmark(
        height=64,
        width=150,
        cell_size=0.01,
        dt=0.003,
        viscosity=0.01,
        u_inf=1.0,
        steps=420,
        station_x=0.7,
        pressure_iters=56,
        diffusion_iters=20,
    )

    assert metrics.rmse_u_over_uinf <= 0.28
    assert metrics.delta99_rel_error <= 0.55
    assert metrics.wall_u_over_uinf <= 0.22
