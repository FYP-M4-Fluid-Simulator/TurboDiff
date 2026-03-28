"""Taylor-Green vortex benchmark for base CFD validation (no SA-RANS)."""

from turbodiff.core.benchmark_validations import run_taylor_green_vortex_benchmark


def main() -> None:
    metrics = run_taylor_green_vortex_benchmark()

    print("=" * 64)
    print("Taylor-Green Vortex Benchmark (Base CFD)")
    print("=" * 64)
    print(f"elapsed time (s)                : {metrics.elapsed_time:.4f}")
    print(f"energy ratio (sim)              : {metrics.energy_ratio_sim:.5f}")
    print(f"energy ratio (analytic)         : {metrics.energy_ratio_analytic:.5f}")
    print(f"energy ratio relative error     : {metrics.energy_ratio_rel_error:.3f}")
    print(f"velocity shape correlation (u)  : {metrics.corr_u:.4f}")
    print(f"velocity shape correlation (v)  : {metrics.corr_v:.4f}")
    print(f"divergence L2                   : {metrics.div_l2:.6f}")
    print(f"divergence max                  : {metrics.div_max:.6f}")

    pass_energy = metrics.energy_ratio_rel_error <= 0.50
    pass_corr = metrics.corr_u >= 0.55 and metrics.corr_v >= 0.55
    pass_div = metrics.div_l2 <= 0.20 and metrics.div_max <= 1.00

    print("-" * 64)
    print(f"PASS energy trend  : {pass_energy}")
    print(f"PASS shape match   : {pass_corr}")
    print(f"PASS incompressible: {pass_div}")
    print(f"OVERALL PASS       : {pass_energy and pass_corr and pass_div}")


if __name__ == "__main__":
    main()
