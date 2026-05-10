import os
import sys
import subprocess


TURBO_DIFF_DIR = "/Users/musab/FYP/TurboDiff"
XFOIL_PATH = "/Users/musab/Xfoil-for-Mac/bin/xfoil"
FORCE_RECREATE = (
    True  # Set to True to delete existing .txt polars and force XFOIL to rerun
)


def run_xfoil(dat_file, re, aoa, results_dir):
    polar_file = dat_file.replace(".dat", ".txt")
    if os.path.exists(polar_file):
        return True

    basename = os.path.basename(dat_file)
    polar_basename = os.path.basename(polar_file)

    commands = f"""LOAD {basename}
PANE
OPER
ITER 300
VISC {re}
PACC
{polar_basename}

ALFA {aoa}
QUIT
"""
    try:
        env = os.environ.copy()
        env["DISPLAY"] = ":0"
        proc = subprocess.Popen(
            XFOIL_PATH,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=results_dir,
        )
        proc.communicate(input=commands, timeout=15)
        return os.path.exists(polar_file)
    except Exception as e:
        print(f"  [XFoil Error] {e}")
        return False


def parse_polar_at_aoa(polar_file, target_aoa):
    if not os.path.exists(polar_file):
        return None

    best_match = None
    min_diff = float("inf")

    with open(polar_file, "r") as f:
        lines = f.readlines()

    start_reading = False
    for line in lines:
        if "alpha" in line and "CL" in line and "CD" in line:
            start_reading = True
            continue
        if start_reading and ("---" in line or not line.strip()):
            continue
        if start_reading:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    alpha, cl, cd = float(parts[0]), float(parts[1]), float(parts[2])
                    diff = abs(alpha - target_aoa)
                    if diff < 0.1 and diff < min_diff:
                        min_diff = diff
                        best_match = (cl, cd)
                except ValueError:
                    continue
    return best_match


def get_max_thickness(dat_file):
    if not os.path.exists(dat_file):
        return None
    with open(dat_file, "r") as f:
        lines = f.readlines()
    coords = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 2:
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not coords:
        return None

    # CST generated airfoils have symmetric x-grids for upper/lower
    # We can group by x and find the spread
    x_map = {}
    for x, y in coords:
        xr = round(x, 4)
        if xr not in x_map:
            x_map[xr] = []
        x_map[xr].append(y)

    max_t = 0
    for xr in x_map:
        if len(x_map[xr]) >= 2:
            t = max(x_map[xr]) - min(x_map[xr])
            if t > max_t:
                max_t = t
    return max_t


def create_summary(suite="s809", output_file=None):
    output_lines = []

    def log(msg):
        print(msg)
        output_lines.append(msg)

    log(f"\n{'='*140}")
    log(f" OPTIMIZATION SUMMARY: {suite.upper()}")
    log(f"{'='*140}")
    header = f"{'Re':<10} | {'AoA':<5} | {'Base L/D':<10} | {'Opt L/D':<10} | {'Imp.':<8} | {'Base CL':<8} | {'Base CD':<8} | {'Opt CL':<8} | {'Opt CD':<8} | {'Base T':<8} | {'Opt T':<8}"
    log(header)
    log("-" * 140)

    results_dir = os.path.join(
        TURBO_DIFF_DIR, "naca_results" if suite == "naca" else "s809_results"
    )

    if FORCE_RECREATE:
        print(
            f"FORCE_RECREATE is True: Purging existing .txt polars in {results_dir}..."
        )
        for f in os.listdir(results_dir):
            if (
                f.endswith(".txt")
                and f != f"{suite}_optimization_summary.txt"
                and f != f"{suite}_final_summary.txt"
            ):
                os.remove(os.path.join(results_dir, f))

    # Naming conventions
    if suite == "s809":
        base_prefix = "tmp_s809_base"
        opt_prefix = "best_s809"
    else:
        base_prefix = "tmp_base"
        opt_prefix = "best_airfoil"

    re_list = [100000, 1000000, 6000000]
    aoa_list = [0, 4, 8] if suite == "naca" else [0, 4]

    for re in re_list:
        for aoa in aoa_list:
            base_polar = os.path.join(
                results_dir, f"{base_prefix}_re_{float(re)}_aoa_{aoa}.txt"
            )
            base_dat = os.path.join(
                results_dir, f"{base_prefix}_re_{float(re)}_aoa_{aoa}.dat"
            )

            opt_polar = os.path.join(
                results_dir, f"{opt_prefix}_re_{float(re)}_aoa_{aoa}.txt"
            )
            opt_dat = os.path.join(
                results_dir, f"{opt_prefix}_re_{float(re)}_aoa_{aoa}.dat"
            )

            # Auto-generate polars if missing
            if not os.path.exists(base_polar) and os.path.exists(base_dat):
                run_xfoil(base_dat, re, aoa, results_dir)
            if not os.path.exists(opt_polar) and os.path.exists(opt_dat):
                run_xfoil(opt_dat, re, aoa, results_dir)

            if not os.path.exists(opt_polar) and not os.path.exists(base_polar):
                continue

            base_perf = parse_polar_at_aoa(base_polar, aoa)
            opt_perf = parse_polar_at_aoa(opt_polar, aoa)

            base_t = get_max_thickness(base_dat)
            opt_t = get_max_thickness(opt_dat)

            re_str = f"{re:.0e}".replace("+0", "")

            if base_perf and opt_perf:
                base_cl, base_cd = base_perf
                opt_cl, opt_cd = opt_perf
                base_ld = base_cl / max(base_cd, 1e-6)
                opt_ld = opt_cl / max(opt_cd, 1e-6)

                if abs(base_ld) < 0.1:
                    imp = 0.0  # Or float('inf') if opt_ld > 0
                else:
                    imp = (opt_ld - base_ld) / base_ld * 100.0

                bt_str = f"{base_t:.3f}" if base_t else "N/A"
                ot_str = f"{opt_t:.3f}" if opt_t else "N/A"

                log(
                    f"{re_str:<10} | {aoa:<5} | {base_ld:<10.2f} | {opt_ld:<10.2f} | {imp:>+7.1f}% | {base_cl:<8.3f} | {base_cd:<8.5f} | {opt_cl:<8.3f} | {opt_cd:<8.5f} | {bt_str:<8} | {ot_str:<8}"
                )
            else:
                log(
                    f"{re_str:<10} | {aoa:<5} | {'N/A':<10} | {'N/A':<10} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}"
                )

    log(f"{'='*140}\n")

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(output_lines))
        print(f"Summary saved to {output_file}")


if __name__ == "__main__":
    target_suite = sys.argv[1].lower() if len(sys.argv) > 1 else "s809"
    out_name = f"{target_suite}_final_summary.txt"
    create_summary(target_suite, output_file=out_name)
