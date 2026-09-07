"""Sweep construction + the multiprocessing work units.

``run_window_analysis`` calls ``algorithms.run_algorithms`` through the module
object (not a ``from .algorithms import run_algorithms`` name binding) so that
``reject_compare`` can swap it out by rebinding
``iaf_compare.algorithms.run_algorithms``. Its bare call to ``compute_spectra``
is swappable the same way, just without needing a module prefix since that
function lives in this same module -- rebinding ``iaf_compare.pipeline.
compute_spectra`` (e.g. to a periodogram-only version, as ``reject_compare``
does) redirects it too.
"""
import numpy as np


from . import algorithms
from .spectra import compute_spectra
from .signal_gen import generate_signal


def build_sweep(fixed, default, sweeps, window_lengths_sec, n_seeds):
    """Build the whole work list for one run.

    Returns ``(conditions, conditions_with_seeds, window_length_groups)``:

    * ``conditions`` -- one-factor-at-a-time sweep: for each param, hold
      everything else at its default and vary that param across its listed
      values (deduplicated). Each signal is exactly ``window_length_sec`` long
      -- one signal == one window == one trial, no sliding.
    * ``conditions_with_seeds`` -- each condition expanded into ``n_seeds``
      ``(global_idx, config, seed)`` items.
    * ``window_length_groups`` -- each condition re-evaluated at every window
      length, sharing ONE parent signal per (condition, seed); each group is
      ``(cond_indices, config, seed, window_lengths_sec)``. Indices continue on
      from ``conditions_with_seeds``.
    """
    seen = set()
    conditions = []
    for param, values in sweeps.items():
        for val in values:
            config = {**fixed, **default, param: val}
            config["signal_length_sec"] = config["window_length_sec"]
            key = tuple(sorted(config.items(), key=lambda x: str(x)))
            if key not in seen:
                seen.add(key)
                conditions.append(config)

    conditions_with_seeds = [
        (i * n_seeds + seed, cfg, seed)
        for i, cfg in enumerate(conditions)
        for seed in range(n_seeds)
    ]

    next_idx = len(conditions_with_seeds)
    window_length_groups = []
    for cfg in conditions:
        for seed in range(n_seeds):
            cond_indices = list(range(next_idx, next_idx + len(window_lengths_sec)))
            next_idx += len(window_lengths_sec)
            window_length_groups.append((cond_indices, cfg, seed, window_lengths_sec))

    return conditions, conditions_with_seeds, window_length_groups


def process_condition(args):
    cond_idx, config, seed = args
    try:
        rng = np.random.default_rng(seed)
        signal, gt_pf = generate_signal(config, rng)
        estimates_per_algo, gt_pf, spectrum_info = run_window_analysis(signal, gt_pf, config)
        stored_config = {**config, "trial_source": "base"}
        # write_results keeps only ONE example spectrum/window per condition
        # (seed 0's -- imap preserves input order, so seed 0 is seen first).
        # Returning the multi-MB arrays for every seed would balloon the
        # accumulated result list to gigabytes at high N_SEEDS, so drop them.
        if seed != 0:
            spectrum_info = None
        return cond_idx, stored_config, gt_pf, estimates_per_algo, spectrum_info
    except Exception as e:
        print(f"Error processing condition\n{config}\n: {e}")
        raise


def process_window_length_condition(args):
    """Like process_condition, but shares ONE parent signal -- generated at
    the longest window_length_sec being compared -- across every variant of
    this (condition, seed), each analyzing a nested sub-window centered on the
    parent's midpoint (the crop in run_window_analysis). This isolates the
    effect of window length itself, instead of confounding it with a fresh
    random draw."""
    cond_indices, base_config, seed, window_lengths_sec = args
    try:
        rng = np.random.default_rng(seed)
        parent_config = {**base_config, "signal_length_sec": max(window_lengths_sec)}
        signal, gt_pf = generate_signal(parent_config, rng)

        results = []
        for cond_idx, window_length_sec in zip(cond_indices, window_lengths_sec):
            config = {**parent_config, "window_length_sec": window_length_sec}
            estimates_per_algo, gt_pf_i, spectrum_info = run_window_analysis(signal, gt_pf, config)
            stored_config = {**config, "trial_source": "window_length_sweep"}
            # See process_condition: only seed 0's example arrays are kept.
            if seed != 0:
                spectrum_info = None
            results.append((cond_idx, stored_config, gt_pf_i, estimates_per_algo, spectrum_info))
        return results
    except Exception as e:
        print(f"Error processing window-length condition group\n{base_config}\n: {e}")
        raise




def run_window_analysis(signal, gt_pf, config):
    """Analyze the signal as a single window covering its full length
    (signals are generated at exactly window_length_sec, so there's nothing to
    slide through), computing the periodogram / Welch / multitaper PSDs and
    running every algorithm on them."""
    fs = config["fs"]
    window_length = int(config["window_length_sec"] * fs)
    if window_length < len(signal):
        window = signal[(len(signal) - window_length) // 2:(len(signal) + window_length) // 2]
    else:
        window = signal
    # window = window * windows.flattop(len(window))  # taper to reduce spectral leakage?

    psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt = compute_spectra(window, window_length, fs, config)

    estimates_per_algo = algorithms.run_algorithms(
        window, psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt, config)

    # Stash the Welch PSD as the per-condition example spectrum (io_hdf5 keeps
    # one per condition, for the spectrum panels / plot_condition_spectra) --
    # it's far smoother than the raw periodogram. Fall back to the periodogram
    # if a caller patched compute_spectra to skip Welch (e.g. reject_compare).
    if psd_welch is not None:
        example_spectrum = (freq_bins_welch, psd_welch)
    else:
        example_spectrum = (freq_bins, psd)

    return estimates_per_algo, gt_pf, (example_spectrum, window)
