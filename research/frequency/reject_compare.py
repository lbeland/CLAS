"""
Compare different `approve_peak` rejection strategies for alpha_fast.
Was IAF_reject_tests.py.

For each strategy in approve_peak_variants.build_strategy_registry():
  1. monkey-patch iaf_compare.algorithms.approve_peak. alpha_fast calls
     approve_peak by name from its own module's globals, so rebinding
     iaf_compare.algorithms.approve_peak is what makes alpha_fast pick it
     up (the call `approve_peak(...)` resolves via algorithms.__dict__ at call
     time). run_window_analysis likewise calls algorithms.run_algorithms
     through the module object, so rebinding that swaps the algorithm set --
     also rebind iaf_compare.pipeline.compute_spectra to a periodogram-only
     version, since only alpha_fast (periodogram-based) ever gets scored here
     and the Welch/multitaper estimates would otherwise be computed on every
     trial for nothing.
  2. run the full with-peak + no-peak condition sweep
  3. pool fp/fn/n across seeds per condition (NOT mean of per-condition
     fail_rate) and store mae/rmse for tp samples
  4. collect one summary row per (strategy, condition) into a single CSV so
     strategies can be compared head-to-head afterward.

Then a single reference-set pass scores the other thesis algorithms (Maximum,
FOOOF, RestingIAF, alpha_fast_mt) once -- none of them vary with the approve_peak
sweep -- into the same rows.

The output is a single raw file, REJECT_DIR/strategy_comparison_raw.csv, with
one row per (strategy, condition). Scoring it into the headline summary, the
LaTeX detection-metrics table and the ROC plot is a separate step -- run
``python plot_reject.py`` afterwards so that (fast) formatting work doesn't
require repeating this (slow) sweep.

Run with: python reject_compare.py
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from iaf_compare import algorithms, pipeline
from iaf_compare.pipeline import process_condition
from iaf_compare.metrics import compute_pooled_metrics
from iaf_compare.paths import REJECT_DIR, ensure_output_dirs
from approve_peak_variants import build_strategy_registry

# The unpatched spectra function (periodogram + Welch + multitaper), captured
# before any worker monkey-patches pipeline.compute_spectra -- the reference-set
# pass restores it so the multitaper PSD is available for alpha_fast_mt.
_full_compute_spectra = pipeline.compute_spectra

# tqdm spins up a background "monitor" thread the first time it's used, and
# that thread persists for the rest of the process. Every Pool() created
# afterward then sees a multi-threaded parent and warns about fork() deadlocks.
# We loop over many strategies, each creating its own Pool(), so without this
# the warning would fire on every strategy after the first. The monitor thread
# isn't used for anything here, so disabling it is safe.
tqdm.monitor_interval = 0

N_SEEDS = 100
N_WORKERS = 5

# The approve_peak strategy that reproduces iaf_compare.algorithms.approve_peak
# verbatim (BIC test on psd_flat, full freq_range, floor_value 0.0). The
# reference-set pass leaves approve_peak at this module default; plot_reject.py
# is where it matters (the only alpha_fast row shown in the detection table).
DEFAULT_BIC_STRATEGY = "bic_only_psd_flat_full"


# Only compare alpha_fast's behavior under each strategy -- fooof and other
# algorithms don't depend on approve_peak, so running them N_strategies times
# would just waste compute on identical results.
def run_algorithms_alpha_fast_only(window, psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt, config):
    return {
        "alpha_fast": algorithms.alpha_fast(psd, freq_bins, config),
    }


# alpha_fast only needs the periodogram -- skip Welch and the multitaper
# eigendecomposition entirely, since run_algorithms_alpha_fast_only above
# never looks at psd_welch/psd_mt anyway.
def compute_spectra_periodogram_only(window, window_length, fs, config):
    freq_bins = np.fft.rfftfreq(window_length, 1 / fs)
    X = np.fft.rfft(window, n=window_length)
    psd = (np.abs(X) ** 2) / (fs * window_length)
    psd[1:-1] *= 2  # Correct for dropping negative freqs in one-sided spectrum (except DC and Nyquist)
    return psd, None, None, freq_bins, None, None


# The reference set doesn't vary with the approve_peak strategy sweep:
# Maximum/FOOOF/RestingIAF ignore approve_peak entirely, and the "alpha_fast_mt"
# row is just alpha_fast on the multitaper PSD -- it calls approve_peak, but the
# table only ever reports the default (inherited unpatched here, ==
# DEFAULT_BIC_STRATEGY). So the set is scored once, as the baseline the
# per-strategy alpha_fast rows are compared against. Uses the full
# periodogram+Welch+multitaper spectra since that row needs the multitaper PSD.
def run_algorithms_reference_set(window, psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt, config):
    return {
        "Maximum":       algorithms.stupid_max(psd_welch, freq_bins_welch, config),
        "FOOOF":         algorithms.fooof(psd_welch, freq_bins_welch, config),
        "RestingIAF":    algorithms.philistine_iaf(psd, freq_bins, config),
        "alpha_fast_mt": algorithms.alpha_fast(psd_mt, freq_mt, config),  # same estimator, multitaper PSD
    }


def build_conditions():
    fixed = {
        "fs":                   10000.0,
        "freq_range":           (0.1, 30.0),
        "alpha_band":           (5, 18),
        "pink_ax_r2":           0.9,
        "aperiodic_ref_power_db": 0.0,
        "f_rotation":           1.0,
    }

    default_with_peak = {
        "peak_freq":            10.32,
        "stationarity":         "constant",   # "constant" | "bursty"
        "aperiodic_exponent":   2.0,
        "n_peaks":              1,
        "peak_width":           0.5,
        "peak_snr_db":          20.0,
        "window_length_sec":    10,
        "noise_lv":             0.5,          # Std dev of the per-bin log10-power noise (gen_noise)
    }
    default_no_peak = {**default_with_peak, "n_peaks": 0}

    sweeps_with_peak = {
        "peak_snr_db":          [0, 5, 10, 20, 50],
        "peak_width":           [0.1, 0.5, 1.0, 2.0],
        "aperiodic_exponent":   [0, 1, 2, 3],
        "peak_freq":            sorted([6, 8, 12, 14] + [default_with_peak["peak_freq"]]),
        "stationarity":         ["constant", "bursty"],
        "noise_lv":             [0, 0.2, 0.5, 1.0],
    }
    sweeps_no_peak = {
        "aperiodic_exponent":   [0, 1, 2, 3],
        "noise_lv":             [0, 0.2, 0.5, 1.0],
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
    algorithms.run_algorithms = run_algorithms_alpha_fast_only
    pipeline.compute_spectra = compute_spectra_periodogram_only


def init_worker_reference_set():
    """Worker init for the single reference-set pass (Maximum, FOOOF, RestingIAF,
    alpha_fast_mt). Restores the full spectra function so the multitaper PSD is
    available for the alpha_fast_mt row, and leaves algorithms.approve_peak at
    its module default (matching DEFAULT_BIC_STRATEGY)."""
    algorithms.run_algorithms = run_algorithms_reference_set
    pipeline.compute_spectra = _full_compute_spectra


def pool_and_score(results, algo_key, row_label):
    """Pool every seed of the same condition, then score. One row per
    (row_label, condition), matching the schema used for the alpha_fast
    strategies so FOOOF/RestingIAF slot into the same table."""
    pooled = {}
    for cond_idx, config, gt, estimates_per_algo, _ in results:
        condition_key = tuple(sorted(config.items(), key=lambda kv: str(kv)))
        samples = pooled.setdefault(condition_key, {"est": [], "gt": [], "config": config,
                                                    "condition_id": cond_idx // N_SEEDS})
        samples["est"].append(float(estimates_per_algo[algo_key]))
        samples["gt"].append(gt)

    rows = []
    for samples in pooled.values():
        config = samples["config"]
        metrics = compute_pooled_metrics(samples["est"], samples["gt"])
        rows.append({
            "strategy": row_label,
            "condition_id": samples["condition_id"],
            "n_peaks": config["n_peaks"],
            "window_length_sec": config["window_length_sec"],
            "peak_snr_db": config["peak_snr_db"],
            "stationarity": config["stationarity"],
            "aperiodic_exponent": config["aperiodic_exponent"],
            "tp": metrics["tp"], "tn": metrics["tn"],
            "fp": metrics["fp"], "fn": metrics["fn"], "n": metrics["n"],
            "mae": metrics["mae"], "rmse": metrics["rmse"],
        })
    return rows


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
        all_rows.extend(pool_and_score(results, "alpha_fast", strategy_name))

    # --- Reference set (Maximum, FOOOF, RestingIAF, alpha_fast_mt): these don't
    # vary with the approve_peak strategy sweep, so run once over the same
    # conditions/seeds and score them into the same table as extra rows.
    print("\n=== Reference set: Maximum + FOOOF + RestingIAF + alpha_fast_mt (scored once) ===")
    with Pool(N_WORKERS, initializer=init_worker_reference_set) as pool:
        ref_results = list(tqdm(
            pool.imap(process_condition, conditions_with_seeds),
            total=len(conditions_with_seeds),
            desc="reference set",
        ))
    for algo_key in ("Maximum", "FOOOF", "RestingIAF", "alpha_fast_mt"):
        all_rows.extend(pool_and_score(ref_results, algo_key, algo_key))

    df = pd.DataFrame(all_rows)
    raw_csv = REJECT_DIR / "strategy_comparison_raw.csv"
    df.to_csv(raw_csv, index=False)
    print(f"\nWrote {len(df)} raw (strategy, condition) rows to {raw_csv}")
    print("Now run `python plot_reject.py` to score it into the summary CSV, "
          "the detection-metrics table and the ROC plot.")


if __name__ == "__main__":
    main()
