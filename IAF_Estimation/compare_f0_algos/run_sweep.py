"""Run the full peak-frequency algorithm sweep and write outputs/iaf_results.h5.

Was IAF_tests.py. The sweep definition now lives in iaf_compare/config.py; the
signal generation, spectral estimation, algorithms, metrics and HDF5 writing
are the corresponding iaf_compare submodules.
"""
from multiprocessing import Pool

from tqdm import tqdm

from iaf_compare import config
from iaf_compare.paths import RESULTS_H5, TABLE_DIR, ensure_output_dirs
from iaf_compare.pipeline import build_sweep, process_condition, process_window_length_condition
from iaf_compare.spectra import precompute_dpss, _init_dpss_cache
from iaf_compare.io_hdf5 import write_results
from iaf_compare.config import save_conditions_latex_table

N_WORKERS = 5  # max(1, cpu_count() - 1)


def main():
    ensure_output_dirs()

    save_conditions_latex_table(
        config.DEFAULT,
        {**config.SWEEPS, "window_length_sec": config.WINDOW_LENGTHS_SEC},
        TABLE_DIR / "conditions_table.tex", fixed=config.FIXED)

    conditions, conditions_with_seeds, window_length_groups = build_sweep(
        config.FIXED, config.DEFAULT, config.SWEEPS,
        config.WINDOW_LENGTHS_SEC, config.N_SEEDS)
    n_base_trials = len(conditions_with_seeds)
    print(f"Total conditions: {len(conditions)}")
    print(f"Total window-length trials: {len(window_length_groups) * len(config.WINDOW_LENGTHS_SEC)}")

    # Precompute every DPSS taper set this run will ever need (one per distinct
    # window_length_sec) in the main process, and hand them to every worker via
    # the Pool initializer -- so each expensive eigendecomposition happens once.
    window_length_secs_needed = ({c["window_length_sec"] for c in conditions}
                                 | set(config.WINDOW_LENGTHS_SEC))
    precomputed_dpss = precompute_dpss(window_length_secs_needed, config.FIXED["fs"])
    print(f"Precomputed {len(precomputed_dpss)} DPSS taper set(s) for window lengths (sec): "
          f"{sorted(window_length_secs_needed)}")

    with Pool(N_WORKERS, initializer=_init_dpss_cache, initargs=(precomputed_dpss,)) as pool:
        base_results = list(tqdm(
            pool.imap(process_condition, conditions_with_seeds),
            total=n_base_trials,
            desc="Processing base conditions",
        ))
        window_length_result_groups = list(tqdm(
            pool.imap(process_window_length_condition, window_length_groups),
            total=len(window_length_groups),
            desc="Processing window-length conditions",
        ))

    all_results = base_results + [row for group in window_length_result_groups for row in group]

    write_results(all_results, RESULTS_H5)
    print(f"Wrote {RESULTS_H5}")


if __name__ == "__main__":
    main()

    # config = {'fs': 10000.0, 'signal_length_sec': 20, 'freq_range': (1.0, 30.0), 'alpha_band': (5, 18),
    #         'pink_ax_r2': 0.8, 'aperiodic_ref_power_db': 0.0, 'f_rotation': 1.0, 'noise_lv': 0.1,
    #         'peak_freq': 14.0, 'stationarity': 'burst', 'aperiodic_exponent': 2, 'n_peaks': 1,
    #         'peak_width': 0.5, 'peak_snr_db': 20.0, 'window_length_sec': 10}
    # from iaf_compare.debug_plots import plot_signal_debug
    # plot_signal_debug(config, n_seconds=20)
    # plot_signal_debug({**config, 'stationarity': 'constant'}, n_seconds=20)
