"""
Main analysis entry point. Run with:
    python -m analysis.main          (from CLAS/)
    python postprocess_results.py    (via the root launcher)
"""

import argparse
import glob
import os
import sys
import traceback
import mne
import yaml
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.loader       import load_processor_signals, analyse_latencies, extract_ground_truth, \
                                  compute_total_latency, find_session_run_dirs
from analysis.f0          import estimate_f0, estimate_f0_with_phase, estimate_f0_modal
from analysis.core         import compute_hilbert_reference, compute_errors, compute_jade_errors, \
                                  _edge_trim_window
from analysis.edf_io       import write_raw_signals_edf, load_runtime, write_analysis_edf
from analysis.runtime_meta import write_runtime_metadata
from analysis.plot         import plot_errors, plot_spectrum, plot_stack, \
                                  get_erp_windows, plot_erp_latency, plot_erp_by_subject, \
                                  load_stim_annotations, plot_pooled_scatter, \
                                  _circ_stats, write_condition_error_table
from analysis.spectrum_analysis import STIM_ONSET_DEGS, CONDITION_LABELS


@dataclass
class RecordingAnalysis:
    """Everything derived from one results folder: the loaded (or cached)
    recording plus the offline f0/Hilbert/error analysis of it. Shared by
    analyse_results() (which additionally plots/exports it) and
    compute_recording_errors() (which just wants .errors)."""
    graph_file:       str
    graph_config:     dict
    fs:               float
    samples:          dict
    ground_truth:     dict
    annotations:      "mne.Annotations | None"
    f0:               float
    snr:              float
    aperiodic_params: np.ndarray
    raw:              np.ndarray
    filtered:         np.ndarray
    hilbert_phase:    np.ndarray
    X_white:          "np.ndarray | None"
    f0_continuous:   np.ndarray
    f0_modal:         np.ndarray
    errors:           list[dict]
    stim_ref:         "np.ndarray | None"


def load_and_analyse(f0: float, results_dir: str, f0_is_truth: bool = False,
                     whiten: bool = False, full_analysis: bool = True) -> "RecordingAnalysis | None":
    """Load one results folder and run the offline analysis on it.

    Loads from cache if available, else the raw .bin files, then computes
    the f0 estimate, Hilbert reference, and phase/f0/stimulus-edge errors.

    Args:
        f0: Centre frequency (Hz) for the pre-Hilbert bandpass; used as a
            fallback unless f0_is_truth is set.
        results_dir: Path to the results folder to load.
        f0_is_truth: Use f0 directly instead of the offline estimate, for
            synthetic runs with a known carrier frequency.
        whiten: Fit the 1/f aperiodic component and use it for spectral
            whitening in compute_hilbert_reference(). Skipped entirely when
            f0_is_truth is also set.
        full_analysis: Run f0 estimation, Hilbert reference and error
            computation. If False, return right after loading (e.g. for ERP
            epoching, which only needs .samples/.fs), leaving those fields
            None.

    Returns:
        A RecordingAnalysis, or None if the folder can't be loaded.
    """
    graph_files = glob.glob(os.path.join(results_dir, "*.yaml"))
    if not graph_files:
        print(f"Error: no graph config (.yaml) found in {results_dir}")
        return None

    graph_file   = graph_files[0]
    raw_edf_path = os.path.join(results_dir, "raw_signals.edf")
    meta_h5_path = os.path.join(results_dir, "runtime_metadata.h5")

    with open(graph_file) as f:
        graph_config = yaml.safe_load(f)

    fs         = graph_config.get("graph", {}).get("defaults", {}).get("fs")
    processors = graph_config.get("graph", {}).get("processors", [])
    if processors is None:
        raise ValueError("No processors found in graph config.")

    if os.path.exists(raw_edf_path) and os.path.exists(meta_h5_path):
        # Cached: skip re-parsing the raw .bin files entirely.
        print(f"Found cached {raw_edf_path} / {meta_h5_path}; skipping raw file loading.")
        samples, ground_truth, annotations = load_runtime(raw_edf_path, meta_h5_path)
    else:
        samples      = load_processor_signals(fs, results_dir, processors)
        ground_truth = extract_ground_truth(samples, graph_config)
        if ground_truth is None:
            print("Error: no source signal found (UDPSource or SimulatedSource). Aborting.")
            return None

        # Persist the raw/runtime signals so future runs can skip this branch
        annotations = load_stim_annotations(results_dir, ground_truth["time"][0])
        write_raw_signals_edf(raw_edf_path, fs, samples, ground_truth, annotations=annotations)
        write_runtime_metadata(meta_h5_path, fs, samples, ground_truth)

    if not full_analysis:
        return RecordingAnalysis(
            graph_file=graph_file, graph_config=graph_config, fs=fs,
            samples=samples, ground_truth=ground_truth, annotations=annotations,
            snr=None, f0=f0, aperiodic_params=None, raw=ground_truth["raw"], filtered=None,
            hilbert_phase=None, X_white=None, f0_continuous=None,
            errors=None, stim_ref=None, f0_modal=None,
        )

    # estimate_f0() returns both f0 and the 1/f aperiodic fit; skip only if
    # neither is wanted (known carrier and whitening disabled)
    snr = None
    if f0_is_truth and not whiten:
        aperiodic_params = None
        print(f"Using provided f0 = {f0:.2f} Hz as ground truth; skipping offline estimation.")
    else:
        f0_windows, aperiodic_params, snr = estimate_f0(ground_truth["raw"], fs)
        if f0_is_truth:
            print(f"Using provided f0 = {f0:.2f} Hz as ground truth "
                  f"(offline f0 estimate ignored).")
        elif np.all(~np.isfinite(f0_windows)):
            print("Could not estimate f0 from ground truth; using default f0 =", f0)
        else:
            f0 = float(np.nanmean(f0_windows))
            print(f"Estimated f0 on 30-second windows: mean={f0:.2f} Hz, all={f0_windows}")
    if not whiten:
        aperiodic_params = None  # disable whitening in compute_hilbert_reference


    # Offline Hilbert reference
    raw = ground_truth["raw"]
    filtered, hilbert_phase, X_white = compute_hilbert_reference(raw, fs, f0, aperiodic_params=aperiodic_params)
    # Needs the raw (unmasked) phase first: NaN edges would propagate through
    # np.unwrap()/cumsum. NaNs its own output's edges internally (see f0.py).
    f0_continuous = estimate_f0_with_phase(hilbert_phase, fs, f0)
    # f0_modal = estimate_f0_modal(raw, fs)

    # NaN hilbert_phase's own edges now, so every series derived from it
    # excludes the filter's edge transients via plain NaN-dropping
    hilbert_phase = hilbert_phase.copy()
    hp_time_s = np.arange(len(hilbert_phase)) / fs
    hp_t_lo, hp_t_hi = _edge_trim_window([{"time_s": hp_time_s}])
    hilbert_phase[(hp_time_s < hp_t_lo) | (hp_time_s > hp_t_hi)] = np.nan

    if ground_truth.get("true_inst_freq") is not None:
        f0_gt = ground_truth["true_inst_freq"]
        if samples.get("FrequencyEstimation") is not None:
            print(f"Online f0 error: {np.nanmean(samples['FrequencyEstimation']['y'] - f0_gt):.2f} Hz +- {np.nanstd(samples['FrequencyEstimation']['y'] - f0_gt):.2f} Hz")
            print(f"Online MAE: {np.nanmean(np.abs(samples['FrequencyEstimation']['y'] - f0_gt)):.2f} Hz")
        print(f"Offline f0 error: {np.nanmean(f0_continuous - f0_gt):.2f} Hz +- {np.nanstd(f0_continuous - f0_gt):.2f} Hz")
        print(f"Offline MAE: {np.nanmean(np.abs(f0_continuous - f0_gt)):.2f} Hz")
    else:
        if samples.get("FrequencyEstimation") is not None:
            online_vs_offline_f0 = samples['FrequencyEstimation']['y'] - f0_continuous
            print(f"Online vs offline f0 error: {np.nanmean(online_vs_offline_f0):.2f} Hz +- {np.nanstd(online_vs_offline_f0):.2f} Hz")
            print(f"Online vs offline MAE: {np.nanmean(np.abs(online_vs_offline_f0)):.2f} Hz")


    # Compute errors (compute_errors also builds a per-onset ECHT phase
    # reference from `filtered` for the "Stim onset error (ECHT)" series)
    start_ts = ground_truth["time"][0]
    errors, stim_ref = compute_errors(samples, ground_truth, hilbert_phase, start_ts, fs,
                                      graph_config)

    # # SimulatedSource ground truth (true_inst_freq known) + short enough for JADE's DTW cost to be feasible
    # if ground_truth["true_inst_freq"] is not None and len(raw) / fs <= 20:
    #     errors += compute_jade_errors(filtered, ground_truth["time"], start_ts, fs,
    #                                    ground_truth["true_phase"], ground_truth["true_inst_freq"],
    #                                    raw=raw, f0=f0)

    return RecordingAnalysis(
        graph_file=graph_file, graph_config=graph_config, fs=fs,
        samples=samples, ground_truth=ground_truth, annotations=annotations,
        snr=snr, f0=f0, aperiodic_params=aperiodic_params, raw=raw, filtered=filtered,
        hilbert_phase=hilbert_phase, X_white=X_white, f0_continuous=f0_continuous,
        errors=errors, stim_ref=stim_ref, f0_modal=None, 
    )


def analyse_results(results_dir: str, f0: float=10.0, show: bool = True,
                    f0_is_truth: bool = False, whiten: bool = False, full_analysis: bool = True) -> None:
    analysis = load_and_analyse(f0, results_dir, f0_is_truth=f0_is_truth, whiten=whiten, full_analysis=full_analysis)
    if analysis is None:
        return

    if full_analysis:
        # Pipeline latency analysis. Reproducible from cache too, since
        # runtime_metadata.h5 preserves source_ts exactly (unlike an EDF round-trip).
        analyse_latencies(analysis.samples, analysis.ground_truth, analysis.graph_config)


        # ERPCLAS-specific: ERP average plot
        if os.path.basename(analysis.graph_file) == "ERPCLAS.yaml":
            windows = get_erp_windows(analysis.fs, analysis.samples, channel=[4,3,27,26])
            plot_erp_latency(windows, analysis.fs)

        # Export analysis outputs to EDF (cheap; rebuilt every run from the cached runtime signals)
        analysis_path = os.path.join(results_dir, "analysis.edf")
        write_analysis_edf(analysis_path, analysis.fs, analysis.ground_truth, analysis.samples,
                            analysis.filtered, analysis.hilbert_phase, analysis.f0_continuous,
                            analysis.stim_ref, annotations=analysis.annotations,
                            whitened=analysis.X_white is not None)

        # Polar error distributions
        plot_errors(analysis.errors)

        # Plot spectrum
        plot_spectrum(analysis.raw, analysis.samples, analysis.filtered, analysis.fs,
                      analysis.aperiodic_params, analysis.X_white)

        # Time-domain panels: stack any subset of "f0"/"phase"/"phase_error"/"signals"
        # e.g. plot_stack(analysis, ["signals", "phase_error"], save_as="sig_vs_err")
        time_range = None # (20.5, 23.5)  # e.g. (17, 18) to zoom in on a few seconds
        # plot_stack(analysis, ["signals", "f0", "phase", "phase_error"], time_range=time_range,
        #            save_as="time_series")
        plot_stack(analysis, ["f0"], time_range=time_range,
                save_as="time_series")

    if show:
        plt.show()
    else:
        plt.close("all")

def find_result_dirs(base_dir: str = "results") -> list[str]:
    """Every directory under base_dir that directly contains a *.yaml graph
    config, i.e. every valid analyse_results() target."""
    return sorted({
        os.path.dirname(p) for p in glob.glob(os.path.join(base_dir, "**", "*.yaml"), recursive=True)
    })


def analyse_all_results(f0: float = 10, base_dir: str = "results", full_analysis: bool = True) -> None:
    """Batch smoke-test: run analyse_results() non-interactively (no
    plt.show()) over every results subfolder under base_dir, keeping going
    past per-folder errors so one bad run doesn't block the rest. Prints a
    full traceback for each failure plus a pass/fail summary at the end."""
    run_dirs = find_result_dirs(base_dir)
    print(f"Found {len(run_dirs)} results folder(s) under {base_dir!r}.")

    failures = []
    for i, results_dir in enumerate(run_dirs, 1):
        print(f"\n{'=' * 80}\n[{i}/{len(run_dirs)}] {results_dir}\n{'=' * 80}")
        try:
            analyse_results(results_dir, f0, show=False, full_analysis=full_analysis)
        except Exception:
            print(f"FAILED: {results_dir}")
            traceback.print_exc()
            failures.append(results_dir)

    print(f"\n{'=' * 80}\n{len(run_dirs) - len(failures)}/{len(run_dirs)} succeeded")
    if failures:
        print("Failed folders:")
        for results_dir in failures:
            print(f"  {results_dir}")
    print("=" * 80)


def compute_recording_errors(f0: float, results_dir: str, f0_is_truth: bool = False,
                             whiten: bool = True) -> list[dict]:
    """Load one results folder and compute its phase/f0/stimulus-edge
    errors, without writing files or plotting. Returns [] if the folder
    can't be loaded or has no source signal. See load_and_analyse() for
    f0_is_truth (take the passed f0 as the known carrier vs. fall back to it)
    and whiten (spectral whitening before the offline bandpass)."""
    analysis = load_and_analyse(f0, results_dir, f0_is_truth=f0_is_truth, whiten=whiten)
    return analysis.errors if analysis is not None else []


def analyse_erp_by_subject(f0: float = 10, base_dir: str = "results", session_glob: str = "CLAS_*",
                           channel: int = 3) -> None:
    """Plot the ERP (default channel 3 = Fz) as one curve per subject, all
    centred on stimulus onset, so per-subject ERP shape/latency can be
    compared at a glance. Subjects are discovered the same way as
    analyse_pooled_errors() (every results/CLAS_* session folder); within
    each, every ERPCLAS.yaml run (normally just the "*_erp_meas_*" one) is
    pooled into that subject's curve."""
    grouped: dict[str, list[str]] = {}
    for run_dir in find_session_run_dirs(base_dir, session_glob, graph_name="ERPCLAS.yaml"):
        grouped.setdefault(os.path.dirname(run_dir), []).append(run_dir)
    print(f"Found {len(grouped)} subject(s) with an ERP measurement under {base_dir}/{session_glob}/*.")

    subject_curves = []
    for session_dir, run_dirs in grouped.items():
        windows = []
        fs = None
        for results_dir in run_dirs:
            print(f"[{os.path.basename(session_dir)}] {results_dir}")
            try:
                analysis = load_and_analyse(f0, results_dir, full_analysis=False)
            except Exception:
                print(f"  FAILED: {results_dir}")
                traceback.print_exc()
                continue
            if analysis is None:
                continue
            fs = analysis.fs
            windows += get_erp_windows(fs, analysis.samples, channel=[channel], average=True)
        if windows:
            subject_curves.append({"name": os.path.basename(session_dir), "windows": windows, "fs": fs})

    if not subject_curves:
        print("No ERP windows collected; nothing to plot.")
        return

    plot_erp_by_subject(subject_curves)
    plt.show()


# Column order for the per-condition error table
_CONDITION_TABLE_COLUMNS = [
    r"$\hat \theta - \theta$", r"$\hat\theta - \theta_{\mathrm{HT}}$",
    r"$\theta_{\mathrm{HT}} - \theta$",
    r"Stim onset ($\theta_{\mathrm{HT}}$)", r"Stim offset ($\theta_{\mathrm{HT}}$)",
    r"Stim onset ($\hat\theta$)", r"Stim offset ($\hat\theta$)",
]

# Row order for the per-condition error table: the four phase-offset
# conditions first, then a rule, then the two P1 (peak/trough) conditions.
_CONDITION_ROW_GROUPS = [[60, 150, 240, 330], [0, 180]]

# Non-circular metrics printed as a single pooled mean +/- SD, not split by condition
_GLOBAL_METRIC_LABELS = ("f0 error", "Stim duration error", "Online vs offline f0 error", "Total latency")


def analyse_pooled_errors(f0: float = 10, names: list[str] = None, base_dir: str = "results", session_glob: str = "CLAS_*") -> None:
    """Pool phase/f0/stimulus-edge errors across every recording and plot/print results.

    Combines every recording under every session folder matching
    session_glob (e.g. results/CLAS_*/<run>) and:
      - Plots pooled polar error-distribution histograms, as
        analyse_results() does per-recording, but combined across all runs.
        No time-window trim is applied: Hilbert-derived series are already
        NaN-masked at their own edges (see load_and_analyse()).
      - Plots per-recording scatter (plot_pooled_scatter) of each
        recording's circular mean +/- SD phase error against its online
        f0-estimate variance and offline alpha-peak SNR.
      - Prints/writes a table of circular mean +/- SD per degree-unit error
        label, one row per stim_onset_deg condition (see
        spectrum_analysis.STIM_ONSET_DEGS/CONDITION_LABELS) and one column
        per label.
      - Prints a plain (non-circular) pooled mean +/- SD for the metrics in
        _GLOBAL_METRIC_LABELS.

    Args:
        f0: Fallback centre frequency (Hz) passed to load_and_analyse() for
            each recording.
        names: If given, only include recordings whose path contains one of
            these strings (case-insensitive).
        base_dir: Root directory to search for session folders.
        session_glob: Glob pattern for session folder names under base_dir.
    """
    run_dirs = find_session_run_dirs(base_dir, session_glob)
    if names is not None:
        run_dirs_selected = [d for d in run_dirs if any(name.lower() in d.lower() for name in names)]
        run_dirs = run_dirs_selected
    print(f"Found {len(run_dirs)} recording(s) under {base_dir}/{session_glob}/*.")

    online_vs_hilbert = r"$\hat\theta - \theta_{\mathrm{HT}}$"  # matches sweep.ONLINE_VS_HILBERT

    pooled: dict[str, dict] = {}
    scatter_records: list[dict] = []
    pooled_by_condition: dict[tuple, list] = {}
    condition_columns: list[str] = []
    global_metrics: dict[str, list] = {}
    for i, results_dir in enumerate(run_dirs, 1):
        print(f"[{i}/{len(run_dirs)}] {results_dir}")
        try:
            analysis = load_and_analyse(f0, results_dir)
        except Exception:
            print(f"  FAILED: {results_dir}")
            traceback.print_exc()
            continue
        if analysis is None:
            continue
        errors = analysis.errors

        # Same metric as load_and_analyse()'s single-run print; appended here so
        # it rides the same _GLOBAL_METRIC_LABELS pooling as f0/stim duration error
        freq_est = analysis.samples.get("FrequencyEstimation")
        if freq_est is not None and analysis.f0_continuous is not None:
            errors = errors + [{
                "label": "Online vs offline f0 error", "unit": "Hz",
                "time_s": (freq_est["x"] - analysis.ground_truth["time"][0]) / 1e6,
                "values": np.asarray(freq_est["y"], dtype=float) - analysis.f0_continuous,
            }]

        # Same total analyse_latencies() prints as "TOTAL" for a single recording
        total_lat = compute_total_latency(analysis.samples, analysis.ground_truth, analysis.graph_config)
        if total_lat is not None:
            n = len(total_lat)
            errors = errors + [{
                "label": "Total latency", "unit": "us",
                "time_s": (analysis.ground_truth["time"][:n] - analysis.ground_truth["time"][0]) / 1e6,
                "values": total_lat.astype(float),
            }]

        condition = (
            analysis.graph_config.get("graph", {})
            .get("processors", {})
            .get("StimControl", {})
            .get("options", {})
            .get("stim_onset_deg")
        )

        for err in errors:
            entry = pooled.setdefault(err["label"], {
                "label": err["label"], "unit": err["unit"], "linestyle": err.get("linestyle", "-"),
                "values": [],
            })
            vals = np.asarray(err["values"], dtype=float)
            if err["unit"] == "degrees":
                vals = vals[~np.isnan(vals)]
                if condition in STIM_ONSET_DEGS and vals.size:
                    pooled_by_condition.setdefault((condition, err["label"]), []).append(vals)
                    if err["label"] not in condition_columns:
                        condition_columns.append(err["label"])
            elif err["label"] in _GLOBAL_METRIC_LABELS:
                trimmed = vals[~np.isnan(vals)]
                if trimmed.size:
                    global_metrics.setdefault(err["label"], []).append(trimmed)
            entry["values"].append(vals)

        # One scatter bullet per recording per plot.py._SCATTER_METRICS metric:
        # circular mean error (y) vs offline SNR (x)
        oh = next((e for e in errors if e["label"] == online_vs_hilbert), None)
        if oh is not None:
            snr_db = (10.0 * np.log10(analysis.snr)
                      if analysis.snr is not None and np.isfinite(analysis.snr) and analysis.snr > 0
                      else np.nan)
            stim_onset_hat_label = r"Stim onset ($\hat\theta$)"
            stim_onset_ht_label  = r"Stim onset ($\theta_{\mathrm{HT}}$)"
            stim_onset_hat = next((e for e in errors if e["label"] == stim_onset_hat_label), None)
            stim_onset_ht  = next((e for e in errors if e["label"] == stim_onset_ht_label), None)
            scatter_records.append({
                "name": os.path.basename(results_dir),
                "phase_vals": np.asarray(oh["values"], dtype=float),
                stim_onset_hat_label: np.asarray(stim_onset_hat["values"], dtype=float)
                                      if stim_onset_hat is not None else np.array([]),
                stim_onset_ht_label: np.asarray(stim_onset_ht["values"], dtype=float)
                                     if stim_onset_ht is not None else np.array([]),
                "snr_db": snr_db,
            })

    if not pooled:
        print("No errors collected; nothing to plot.")
        return

    merged_errors = [
        {"label": e["label"], "unit": e["unit"], "linestyle": e["linestyle"],
         "values": np.concatenate(e["values"])}
        for e in pooled.values()
    ]

    plot_errors(merged_errors)
    plot_pooled_scatter(scatter_records)

    if pooled_by_condition:
        condition_stats: dict[int, dict[str, tuple[float, float, float]]] = {}
        for (condition, label), chunks in pooled_by_condition.items():
            mu_u, sd_u, plv_u, _ = _circ_stats(np.radians(np.concatenate(chunks)))
            condition_stats.setdefault(condition, {})[label] = (np.degrees(mu_u), np.degrees(sd_u), plv_u)
        columns_ordered = [c for c in _CONDITION_TABLE_COLUMNS if c in condition_columns] + \
                          sorted(c for c in condition_columns if c not in _CONDITION_TABLE_COLUMNS)
        write_condition_error_table(condition_stats, columns_ordered, _CONDITION_ROW_GROUPS, CONDITION_LABELS)
    else:
        print("No recording matched a known stim_onset_deg condition; skipping condition table.")

    # Plain pooled mean +/- SD across every recording, printed to console
    for label, chunks in global_metrics.items():
        vals = np.concatenate(chunks)
        unit = pooled[label]["unit"]
        print(f"{label}: {np.mean(vals):.3f} +- {np.std(vals):.3f} {unit} "
              f"(n={vals.size} samples, {len(chunks)} recording(s))")
        if label == "Online vs offline f0 error":
            print(f"Online vs offline MAE: {np.mean(np.abs(vals)):.3f} {unit}")
        if label == "Stim duration error":
            print(f"Stim duration MAE: {np.mean(np.abs(vals)):.3f} {unit}")

    plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline CLAS analysis entry point -- pick which of the analyse_*() "
                     "functions to run instead of (un)commenting them below.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    all_p = subparsers.add_parser(
        "all", help="analyse_all_results(): analyze every results/ subfolder to generate .edf and .h5 files.")
    all_p.add_argument("--base-dir", default="results",
                        help="Base directory to search for results folders (default: results).")
    all_p.add_argument("--f0", type=float, default=10.0,
                        help="Fallback f0 (Hz) if offline estimation fails (default: 10.0).")
    all_p.add_argument("--no-full-analysis", dest="full_analysis", action="store_false",
                        help="Skip f0/Hilbert/error analysis; just load (and cache) each folder.")

    single_p = subparsers.add_parser(
        "single", help="analyse_results(): full analysis + plots for one results folder.")
    single_p.add_argument("results_dir", nargs="?", default="_last_run",
                           help="Results folder to analyse (default: _last_run).")
    single_p.add_argument("--f0", type=float, default=10.0,
                           help="Centre frequency (Hz) for the pre-Hilbert bandpass; fallback "
                                "unless --f0-is-truth is set (default: 10.0).")
    single_p.add_argument("--f0-is-truth", action="store_true",
                           help="Use --f0 directly instead of re-estimating it offline.")
    single_p.add_argument("--whiten", action="store_true",
                           help="Apply 1/f spectral whitening before the offline bandpass.")
    single_p.add_argument("--no-show", dest="show", action="store_false",
                           help="Don't call plt.show() at the end (just close the figures).")
    single_p.add_argument("--no-full-analysis", dest="full_analysis", action="store_false",
                           help="Skip f0/Hilbert/error analysis and plotting; just load the folder.")

    pooled_p = subparsers.add_parser(
        "pooled", help="analyse_pooled_errors(): pool errors across every session's recordings.")
    pooled_p.add_argument("--f0", type=float, default=10.0)
    pooled_p.add_argument("--base-dir", default="results")
    pooled_p.add_argument("--session-glob", default="CLAS_*",
                           help="Glob (under --base-dir) matching session folders (default: CLAS_*).")
    pooled_p.add_argument("--names", nargs="*", default=None,
                           help="Only include recordings whose path contains one of these "
                                "substrings (case-insensitive), e.g. --names victor.")

    erp_p = subparsers.add_parser(
        "erp", help="analyse_erp_by_subject(): plot one ERP curve per subject.")
    erp_p.add_argument("--f0", type=float, default=10.0)
    erp_p.add_argument("--base-dir", default="results")
    erp_p.add_argument("--session-glob", default="CLAS_*")
    erp_p.add_argument("--channel", type=int, default=3,
                        help="Channel index to average for the ERP (default: 3 = Fz).")

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.command == "all":
        analyse_all_results(f0=args.f0, base_dir=args.base_dir, full_analysis=args.full_analysis)
    elif args.command == "single":
        analyse_results(results_dir=args.results_dir, f0=args.f0, show=args.show,
                         f0_is_truth=args.f0_is_truth, whiten=args.whiten,
                         full_analysis=args.full_analysis)
    elif args.command == "pooled":
        analyse_pooled_errors(f0=args.f0, names=args.names, base_dir=args.base_dir,
                              session_glob=args.session_glob)
    elif args.command == "erp":
        analyse_erp_by_subject(f0=args.f0, base_dir=args.base_dir,
                               session_glob=args.session_glob, channel=args.channel)

