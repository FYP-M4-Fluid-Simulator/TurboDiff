"""Blasius flat-plate benchmark for base CFD validation (no SA-RANS)."""

from turbodiff.core.benchmark_validations import run_blasius_boundary_layer_benchmark


def main() -> None:
    metrics = run_blasius_boundary_layer_benchmark()

    print("=" * 64)
    print("Blasius Boundary Layer Benchmark (Base CFD)")
    print("=" * 64)
    print(f"x station (m)                  : {metrics.station_x:.4f}")
    print(f"RMSE[u/Uinf] vs Blasius        : {metrics.rmse_u_over_uinf:.4f}")
    print(f"delta99 sim (m)                : {metrics.delta99_sim:.5f}")
    print(f"delta99 ref (m)                : {metrics.delta99_ref:.5f}")
    print(f"delta99 relative error         : {metrics.delta99_rel_error:.3f}")
    print(f"u/Uinf at first cell from wall : {metrics.wall_u_over_uinf:.4f}")

    pass_rmse = metrics.rmse_u_over_uinf <= 0.25
    pass_delta = metrics.delta99_rel_error <= 0.55
    pass_wall = metrics.wall_u_over_uinf <= 0.20

    print("-" * 64)
    print(f"PASS profile RMSE : {pass_rmse}")
    print(f"PASS delta99      : {pass_delta}")
    print(f"PASS wall no-slip : {pass_wall}")
    print(f"OVERALL PASS      : {pass_rmse and pass_delta and pass_wall}")


if __name__ == "__main__":
    main()
