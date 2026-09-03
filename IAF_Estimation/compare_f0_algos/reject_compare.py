"""
Compare different `approve_peak` rejection strategies for combine_simple.
Was IAF_reject_tests.py.

For each strategy in approve_peak_variants.build_strategy_registry():
  1. monkey-patch iaf_compare.algorithms.approve_peak. combine_simple calls
     approve_peak by name from its own module's globals, so rebinding
     iaf_compare.algorithms.approve_peak is what makes combine_simple pick it
     up (the call `approve_peak(...)` resolves via algorithms.__dict__ at call
     time). run_window_analysis likewise calls algorithms.run_algorithms
     through the module object, so rebinding that swaps the algorithm set.
  2. run the full with-peak + no-peak condition sweep
  3. pool fp/fn/n across seeds per condition (NOT mean of per-condition
     fail_rate) and store mae/rmse for tp samples
  4. collect one summary row per (strategy, condition) into a single CSV so
     strategies can be compared head-to-head afterward.

Run with: python reject_compare.py
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from iaf_compare import algorithms
from iaf_compare.pipeline import process_condition
from iaf_compare.metrics import compute_pooled_metrics
from iaf_compare.paths import REJECT_DIR, ensure_output_dirs
from approve_peak_variants import build_strategy_registry

# tqdm spins up a background "monitor" thread the first time it's used, and
# that thread persists for the rest of the process. Every Pool() created
# afterward then sees a multi-threaded parent and warns about fork() deadlocks.
# We loop over many strategies, each creating its own Pool(), so without this
# the warning would fire on every strategy after the first. The monitor thread
# isn't used for anything here, so disabling it is safe.
tqdm.monitor_interval = 0

N_SEEDS = 20
N_WORKERS = 5


# Only compare combine_simple's behavior under each strategy -- fooof and other
# algorithms don't depend on approve_peak, so running them N_strategies times
# would just waste compute on identical results.
def run_algorithms_combine_simple_only(window, psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt, config):
    return {
        "combine_simple": algorithms.combine_simple(psd, freq_bins, config),
    }


def build_conditions():
    fixed = {
        "fs":                   10000.0,
        "freq_range":           (0.01, 30.0),
        "alpha_band":           (5, 18),
        "pink_ax_r2":           0.9,
        "aperiodic_ref_power_db": 0.0,
        "f_rotation":           1.0,
    }

    default_with_peak = {
        "peak_freq":            14.0,
        "stationarity":         "constant",   # "constant" | "burst"
        "aperiodic_exponent":   2.0,
        "n_peaks":              1,
        "peak_width":           0.5,  # was "peak_bw" -- generate_signal reads "peak_width"
        "peak_snr_db":          20.0,
        "window_length_sec":    10,
        "noise_lv":             0.2,          # Std dev of the per-bin log10-power noise (gen_noise)
    }
    default_no_peak = {**default_with_peak, "n_peaks": 0}

    sweeps_with_peak = {
        "peak_snr_db":          [0, 5, 10],
        "peak_width":           [0.1, 0.5, 1.0, 2.0, 4.0],  # was "peak_bw"
        "aperiodic_exponent":   [0, 1, 2, 3],
        "peak_freq":            sorted([6, 8, 12, 14] + [default_with_peak["peak_freq"]]),
        "stationarity":         ["constant", "burst"],
        "noise_lv":             [0, 0.2, 1.0],
    }
    sweeps_no_peak = {
        "aperiodic_exponent":   [0, 1, 2, 3],
        "noise_lv":             [0, 0.2, 1.0],
    }

    seen = set()
    conditions = []

    for sweeps, default in ((sweeps_with_peak, default_with_peak),
                            (sweeps_no_peak, default_no_peak)):
        for param, values in sweeps.items():
            for val in values:
                config = {**fixed, **default, param: val}
                # Generate each signal at exactly window_length_sec -- one
                # signal == one window == one trial, no sliding.
                config["signal_length_sec"] = config["window_length_sec"]
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
    algorithms.approve_peak = registry[strategy_name]
    algorithms.run_algorithms = run_algorithms_combine_simple_only


def main():
    ensure_output_dirs()
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

        # Pool every seed of the same condition before scoring -- fp/fn/mae/
        # rmse are only meaningful pooled over all seeds, not per single trial.
        pooled = {}
        for cond_idx, config, gt, estimates_per_algo, _ in results:
            condition_key = tuple(sorted(config.items(), key=lambda kv: str(kv)))
            samples = pooled.setdefault(condition_key, {"est": [], "gt": [], "config": config,
                                                        "condition_id": cond_idx // N_SEEDS})
            samples["est"].append(float(estimates_per_algo["combine_simple"]))
            samples["gt"].append(gt)

        for samples in pooled.values():
            config = samples["config"]
            metrics = compute_pooled_metrics(samples["est"], samples["gt"])
            all_rows.append({
                "strategy": strategy_name,
                "condition_id": samples["condition_id"],
                "n_peaks": config["n_peaks"],
                "window_length_sec": config["window_length_sec"],
                "peak_snr_db": config["peak_snr_db"],
                "stationarity": config["stationarity"],
                "aperiodic_exponent": config["aperiodic_exponent"],
                "fp": metrics["fp"], "fn": metrics["fn"], "n": metrics["n"],
                "mae": metrics["mae"], "rmse": metrics["rmse"],
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(REJECT_DIR / "strategy_comparison_raw.csv", index=False)

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
        #                                   conditions with n_peaks>1 if one of
        #                                   several peaks is spuriously found
        #                                   where none exists in that window --
        #                                   keep separate from fn
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
    summary.to_csv(REJECT_DIR / "strategy_comparison_summary.csv", index=False)
    print("\n=== Summary (sorted by false positive rate) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
