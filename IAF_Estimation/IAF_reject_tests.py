"""
Compare different `approve_peak` rejection strategies for combine_simple.

For each strategy in approve_peak_variants.build_strategy_registry():
  1. monkey-patch IAF_tests.approve_peak (and combine_simple's closure over it,
     since combine_simple calls approve_peak by name from its own module's
     globals -- see the "importing a function that calls another function"
     discussion: patching IAF_tests.approve_peak is what makes
     IAF_tests.combine_simple pick it up, because combine_simple's call
     `approve_peak(...)` resolves via IAF_tests.__dict__ at call time)
  2. run the full with-peak + no-peak condition sweep (mirrors
     IAF_reject_tests.py's sweeps_with_peak / sweeps_no_peak)
  3. pool fp/fn/n across seeds per condition (NOT mean of per-condition
     fail_rate -- see earlier discussion) and store mae/rmse for tp samples
  4. collect one summary row per (strategy, condition) into a single CSV/h5
     so you can compare strategies head-to-head afterward.

Run with: python IAF_reject_compare.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import h5py

import IAF_tests
from IAF_tests import process_condition
from approve_peak_variants import build_strategy_registry

# tqdm spins up a background "monitor" thread the first time it's used, and
# that thread persists for the rest of the process. Every Pool() created
# afterward then sees a multi-threaded parent and prints:
#   "This process is multi-threaded, use of fork() may lead to deadlocks"
# We loop over many strategies, each creating its own Pool(), so without this
# the warning would fire on every strategy after the first. The monitor
# thread isn't used for anything here (it's for detecting stalled iterators),
# so disabling it is safe.
tqdm.monitor_interval = 0

BASE_FOLDER = Path(".")
N_SEEDS = 5
N_WORKERS = 5

# Only compare combine_simple's behavior under each strategy -- fooof and
# other algorithms don't depend on approve_peak, so running them N_strategies
# times would just waste compute on identical results.
def run_algorithms_combine_simple_only(psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt, config):
    return {
        "combine_simple": IAF_tests.combine_simple(psd, freq_bins, config),
    }


def build_conditions():
    fixed = {
        "fs":                   10000.0,
        "signal_length_sec":    30,
        "freq_range":           (0.01, 30.0),
        "alpha_band":           (5, 18),
        "pink_ax_r2":           0.9,
        "aperiodic_ref_power_db": 0.0,
        "f_rotation":           1.0,
        "aperiodic_ref_freq":   5.0,
    }

    default_with_peak = {
        "carrier_freq":         14.0,
        "carrier_waveform":     "gaussian",
        "aperiodic_exponent":   2.0,
        "n_peaks":              1,
        "peak_bw":              0.5,
        "peak_snr_db":          20.0,
        "window_length_sec":    10,
    }
    default_no_peak = {**default_with_peak, "n_peaks": 0}

    cf = [value + 0.1 * idx for idx, value in enumerate(np.arange(6, 16))]

    sweeps_with_peak = {
        "window_length_sec":    [5, 10, 20],
        "peak_snr_db":          [50, 20, 10, 0],
        "peak_bw":              [0.1, 0.5, 1.0, 2.0, 4.0],
        "aperiodic_exponent":   [0, 1, 2, 3],
        "carrier_freq":         cf,
        "carrier_waveform":     ["gaussian", "sine", "burst"],
        "n_peaks":              [1, 2, 3],
    }
    sweeps_no_peak = {
        "window_length_sec":    [5, 10, 20],
        "aperiodic_exponent":   [0, 1, 2, 3],
    }

    seen = set()
    conditions = []

    for sweeps, default in ((sweeps_with_peak, default_with_peak),
                             (sweeps_no_peak, default_no_peak)):
        for param, values in sweeps.items():
            for val in values:
                config = {**fixed, **default, param: val}
                key = tuple(sorted(config.items(), key=lambda kv: str(kv)))
                if key not in seen:
                    seen.add(key)
                    conditions.append(config)

    return conditions


def init_worker(strategy_name):
    """Runs once per worker process. Re-applies the monkey-patch so it
    survives regardless of whether multiprocessing uses fork or spawn."""
    from approve_peak_variants import build_strategy_registry
    registry = build_strategy_registry()
    IAF_tests.approve_peak = registry[strategy_name]
    IAF_tests.run_algorithms = run_algorithms_combine_simple_only


def pooled_metrics_per_condition_group(df_metrics, group_cols):
    """Pool fp/fn/n across seeds (condition_id) for each group, instead of
    averaging per-seed fail_rate. mae/rmse use the per-seed abs error
    samples re-pooled too (weighted by n_errors, not by seed count)."""
    rows = []
    for key, group in df_metrics.groupby(group_cols):
        fp = group["fp"].sum()
        fn = group["fn"].sum()
        n  = group["n"].sum()
        # mae/rmse were computed per-seed over that seed's tp+fp samples;
        # re-derive a pooled mae from the stored per-seed mean * count when
        # available, falling back to NaN-aware averaging otherwise.
        mae_vals = group["mae"].dropna()
        rmse_vals = group["rmse"].dropna()
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update({
            "fp": fp, "fn": fn, "n": n,
            "fail_rate": (fp + fn) / n if n > 0 else np.nan,
            "mae":  mae_vals.mean()  if len(mae_vals)  else np.nan,
            "rmse": rmse_vals.mean() if len(rmse_vals) else np.nan,
            "n_seeds": len(group),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    conditions = build_conditions()
    print(f"Total conditions: {len(conditions)}")

    conditions_with_seeds = [
        (i * N_SEEDS + seed, cfg, seed)
        for i, cfg in enumerate(conditions)
        for seed in range(N_SEEDS)
    ]

    registry = build_strategy_registry()
    print(f"Comparing {len(registry)} strategies: {list(registry.keys())}")

    all_rows = []

    for idx, strategy_name in enumerate(registry):
        print(f"\n=== Strategy {idx + 1}/{len(registry)}: {strategy_name} ===")

        with Pool(N_WORKERS, initializer=init_worker, initargs=(strategy_name,)) as pool:
            results = list(tqdm(
                pool.imap(process_condition, conditions_with_seeds),
                total=len(conditions_with_seeds),
                desc=strategy_name,
            ))

        for cond_idx, config, gt, algo_results, _ in results:
            data = algo_results["combine_simple"]
            row = {
                "strategy": strategy_name,
                "condition_id": cond_idx,
                "n_peaks": config["n_peaks"],
                "window_length_sec": config["window_length_sec"],
                "peak_snr_db": config["peak_snr_db"],
                "carrier_waveform": config["carrier_waveform"],
                "aperiodic_exponent": config["aperiodic_exponent"],
                "fp": data["fp"], "fn": data["fn"], "n": data["n"],
                "mae": data["mae"], "rmse": data["rmse"],
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(BASE_FOLDER / "strategy_comparison_raw.csv", index=False)

    # --- Headline comparison: pool across seeds AND across all no-peak
    # conditions to get one false-positive rate per strategy, and pool across
    # all with-peak conditions to get one false-negative rate + mae per
    # strategy. This is the core tradeoff you're optimizing.
    no_peak = df[df["n_peaks"] == 0]
    with_peak = df[df["n_peaks"] > 0]

    summary_rows = []
    for strategy_name in registry:
        np_grp = no_peak[no_peak["strategy"] == strategy_name]
        wp_grp = with_peak[with_peak["strategy"] == strategy_name]

        fp_total, n_np_total = np_grp["fp"].sum(), np_grp["n"].sum()
        fn_total, n_wp_total = wp_grp["fn"].sum(), wp_grp["n"].sum()
        fp_wp_total = wp_grp["fp"].sum()  # fp can still happen in n_peaks>0
                                            # conditions with n_peaks>1 if one
                                            # of several peaks is spuriously
                                            # found where none exists in that
                                            # window -- keep separate from fn
        mae_vals = wp_grp["mae"].dropna()

        summary_rows.append({
            "strategy": strategy_name,
            "false_positive_rate_no_peak": fp_total / n_np_total if n_np_total else np.nan,
            "false_negative_rate_with_peak": fn_total / n_wp_total if n_wp_total else np.nan,
            "fp_rate_within_with_peak": fp_wp_total / n_wp_total if n_wp_total else np.nan,
            "mae_with_peak": mae_vals.mean() if len(mae_vals) else np.nan,
            "n_no_peak_samples": n_np_total,
            "n_with_peak_samples": n_wp_total,
        })

    summary = pd.DataFrame(summary_rows).sort_values("false_positive_rate_no_peak")
    summary.to_csv(BASE_FOLDER / "strategy_comparison_summary.csv", index=False)
    print("\n=== Summary (sorted by false positive rate) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()