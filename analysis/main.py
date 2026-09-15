"""
Main analysis entry point. Run with:
    python -m analysis.main          (from CLAS/)
    python postprocess_results.py    (via the root launcher)
"""

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
from analysis.core         import compute_hilbert_reference, compute_errors, compute_jade_errors
from analysis.edf_io       import write_raw_signals_edf, load_runtime, write_analysis_edf
from analysis.runtime_meta import write_runtime_metadata
from analysis.plot         import plot_errors, plot_spectrum, plot_stack, \
                                  get_erp_windows, plot_erp_latency, plot_erp_by_subject, \
                                  load_stim_annotations, plot_pooled_scatter, _edge_trim_window, \
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
    """Load one results folder (from cache if available, else the raw .bin
    files) and run the offline analysis on it: f0 estimate, Hilbert
    reference, phase/f0/stimulus-edge errors. Returns None if the folder
    can't be loaded (no graph config / no source signal).

    f0 sets the centre frequency of the pre-Hilbert bandpass. By default it
    is only a fallback: f0 is re-estimated offline from the recording and the
    passed value is used only if that estimation yields nothing. With
    f0_is_truth=True the passed f0 is used directly (the offline f0 estimate
    is not adopted) -- for synthetic runs whose carrier frequency is known
    exactly.

    whiten (default False): still run the 1/f aperiodic fit and pass it to
    compute_hilbert_reference() for spectral whitening before the bandpass,
    regardless of f0_is_truth. Set whiten=False to skip that whitening; with
    f0_is_truth=True as well, the offline aperiodic/f0 fit is skipped entirely.

    full_analysis (default True): run the offline f0 estimation, Hilbert
    reference and error computation below. Set to False to skip all of that
    and return right after loading -- e.g. for ERP epoching, which only
    needs .samples/.fs -- leaving f0/aperiodic_params/raw/filtered/
    hilbert_phase/X_white/f0_continuous/errors/stim_ref as None (f0 keeps
    the passed-in value)."""
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

    # Centre frequency f0 for the pre-Hilbert bandpass + 1/f aperiodic fit for
    # spectral whitening. estimate_f0() returns both; run it unless neither is
    # wanted (known carrier and whitening disabled).
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
    # estimate_f0_with_phase() needs the raw (unmasked) phase -- its internal
    # np.unwrap()/cumsum would propagate NaN across the whole array if fed
    # edges that are already NaN -- so it runs first, on the clean signal; it
    # NaNs its own output's edges internally (see f0.py).
    f0_continuous = estimate_f0_with_phase(hilbert_phase, fs, f0)
    # f0_modal = estimate_f0_modal(raw, fs)

    # NaN the same leading/trailing window on hilbert_phase itself now that
    # f0_continuous no longer needs the unmasked version. Every error series
    # derived from hilbert_phase (compute_errors(), the online-vs-Hilbert
    # scatter, sweep.py's per-run stats, ...) then excludes exactly the
    # Hilbert filter's edge transients via ordinary NaN-dropping, instead of
    # every consumer re-applying its own blanket time-window trim regardless
    # of whether that series actually involves the Hilbert phase at all.
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
                    f0_is_truth: bool = False, whiten: bool = False) -> None:
    analysis = load_and_analyse(f0, results_dir, f0_is_truth=f0_is_truth, whiten=whiten)
    if analysis is None:
        return

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
                        analysis.stim_ref, annotations=analysis.annotations)


    # Polar error distributions
    plot_errors(analysis.errors)

    # Plot spectrum
    plot_spectrum(analysis.raw, analysis.samples, analysis.filtered, analysis.fs,
                  analysis.aperiodic_params, analysis.X_white)

    # Time-domain panels. Stack any subset of "f0" / "phase" / "phase_error" /
    # "signals" sharing a common time axis; pass a different panel list here
    # for other use cases, e.g.
    #   plot_stack(analysis, ["signals", "phase_error"], save_as="sig_vs_err")
    #   plot_stack(analysis, ["signals", "f0", "phase", "phase_error"],
    #              time_range=time_range, save_as="everything")
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
    config -- i.e. every valid analyse_results() target, including ones
    nested under a session folder like results/CLAS_*/*."""
    return sorted({
        os.path.dirname(p) for p in glob.glob(os.path.join(base_dir, "**", "*.yaml"), recursive=True)
    })


def analyse_all_results(f0: float = 10, base_dir: str = "results") -> None:
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
            analyse_results(results_dir, f0, show=False)
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


# Column order for the per-condition error table below -- the degree-unit
# error labels compute_errors() can produce, in a fixed, readable order
# (phase error first, then onset/offset pairs) rather than whatever order
# recordings happen to be discovered in.
_CONDITION_TABLE_COLUMNS = [
    r"$\hat \theta - \theta$", r"$\hat \theta - \theta_{\mathrm{HT}}$",
    r"$\theta_{\mathrm{HT}} - \theta$", r"$\hat\theta - \theta_{\mathrm{HT}}$",
    r"Stim onset ($\theta_{\mathrm{HT}}$)", r"Stim offset ($\theta_{\mathrm{HT}}$)",
    r"Stim onset ($\hat\theta$)", r"Stim offset ($\hat\theta$)",
]

# Row order for the per-condition error table: the four phase-offset
# conditions first, then a rule, then the two P1 (peak/trough) conditions.
_CONDITION_ROW_GROUPS = [[60, 150, 240, 330], [0, 180]]

# Non-circular metrics (unrelated to stim_onset_deg condition) just printed
# as a single pooled mean +/- SD across every recording -- not part of the
# per-condition table above.
_GLOBAL_METRIC_LABELS = ("f0 error", "Stim duration error", "Online vs offline f0 error", "Total latency")


def analyse_pooled_errors(f0: float = 10, names: list[str] = None, base_dir: str = "results", session_glob: str = "CLAS_*") -> None:
    """Pool phase/f0/stimulus-edge errors across every recording nested
    under every session folder matching session_glob (e.g. every
    results/CLAS_*/<run>), and plot the polar error-distribution histograms
    exactly like analyse_results() does for a single recording -- but
    combined across all of them, one polar distribution per error label.
    No blanket per-recording time-window trim is applied before pooling:
    hilbert_phase/f0_continuous are already NaN-masked at their own edges
    (see load_and_analyse()), so a series derived from them already excludes
    those samples, while a series that doesn't touch the Hilbert reference
    (ground-truth f0 error, stim duration error, total latency, an online
    estimate scored against true_phase) is pooled in full.

    Also produces two per-recording scatter plots (plot_pooled_scatter): the
    circular mean +/- SD of each recording's online-vs-Hilbert phase error on
    the x-axis, against that recording's online (Kalman) f0-estimate variance
    and its offline alpha-peak SNR (dB) on the y-axis.

    Also writes (and prints) a table with one row per stimulus-onset
    condition (StimControl.options.stim_onset_deg in each recording's graph
    yaml -- see spectrum_analysis.STIM_ONSET_DEGS/CONDITION_LABELS for the 6
    conditions) and one column per degree-unit error label, each cell the
    circular mean +/- SD of that metric pooled across every recording run
    under that condition.

    And prints a plain pooled mean +/- SD (ordinary, not circular; not split
    by condition) for f0 error [Hz] and stim duration error [ms] across every
    recording -- see _GLOBAL_METRIC_LABELS."""
    run_dirs = find_session_run_dirs(base_dir, session_glob)
    if names is not None:
        run_dirs_selected = [d for d in run_dirs if any(name.lower() in d.lower() for name in names)]
        run_dirs = run_dirs_selected
    print(f"Found {len(run_dirs)} recording(s) under {base_dir}/{session_glob}/*.")

    # The online-vs-Hilbert phase-error label, ignoring the "\hat \theta" vs
    # "\hat\theta" spacing difference between compute_errors()'s primary series
    # (no true_phase) and its dedicated synthetic-run series.
    online_vs_hilbert = r"$\hat\theta-\theta_{\mathrm{HT}}$"

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

        # Same metric as the single-run "Online vs offline f0 error" print in
        # load_and_analyse(); appended here (not in compute_errors()) so it
        # rides the same edge-trimmed _GLOBAL_METRIC_LABELS pooling as f0/stim
        # duration error below instead of a separate accumulator.
        freq_est = analysis.samples.get("FrequencyEstimation")
        if freq_est is not None and analysis.f0_continuous is not None:
            errors = errors + [{
                "label": "Online vs offline f0 error", "unit": "Hz",
                "time_s": (freq_est["x"] - analysis.ground_truth["time"][0]) / 1e6,
                "values": np.asarray(freq_est["y"], dtype=float) - analysis.f0_continuous,
            }]

        # End-to-end pipeline latency (source -> last processor), same number
        # analyse_latencies() prints as "TOTAL" for a single recording --
        # computed once by compute_total_latency() and pooled here the same
        # way as the other _GLOBAL_METRIC_LABELS series.
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

        # Edge transients are no longer trimmed by a blanket per-recording
        # time window here: hilbert_phase and f0_continuous already carry NaN
        # on their own leading/trailing edges (see load_and_analyse()), so
        # every series actually derived from them already excludes those
        # samples, while series that don't touch the Hilbert reference at all
        # (ground-truth f0 error, stim duration error, total latency, an
        # online estimate scored against true_phase) are left untouched
        # instead of being needlessly cut at both ends.
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

        # One scatter bullet per recording: online-vs-Hilbert phase error (x)
        # vs online f0-estimate variance and offline SNR (y).
        oh = next((e for e in errors
                   if e["label"].replace(" ", "") == online_vs_hilbert), None)
        if oh is not None:
            freq_est = analysis.samples.get("FrequencyEstimation")
            f0_var = (float(np.nanvar(np.asarray(freq_est["y"], dtype=float)))
                      if freq_est is not None and len(freq_est["y"]) else np.nan)
            snr_db = (10.0 * np.log10(analysis.snr)
                      if analysis.snr is not None and np.isfinite(analysis.snr) and analysis.snr > 0
                      else np.nan)
            scatter_records.append({
                "name": os.path.basename(results_dir),
                "phase_vals": np.asarray(oh["values"], dtype=float),
                "f0_var": f0_var, "snr_db": snr_db,
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

    # Simple pooled mean +/- SD across every recording (not per-condition,
    # not circular) for f0/stim-duration/online-vs-offline-f0 error -- just a
    # console print.
    for label, chunks in global_metrics.items():
        vals = np.concatenate(chunks)
        unit = pooled[label]["unit"]
        print(f"{label}: {np.mean(vals):.3f} +- {np.std(vals):.3f} {unit} "
              f"(n={vals.size} samples, {len(chunks)} recording(s))")
        if label == "Online vs offline f0 error":
            print(f"Online vs offline MAE: {np.mean(np.abs(vals)):.3f} {unit}")

    plt.show()


if __name__ == "__main__":
    # analyse_all_results()

    # analyse_results(results_dir="_last_run") #, f0_is_truth=True)
    # analyse_results(results_dir="results/CLAS_steffen/steffen_60_20260819_155602") #, f0_is_truth=True)
    
    # analyse_results(results_dir="results/snr_sweep/ecHTtests/snr-10_pink_20260909_100011", f0_is_truth=True)

    analyse_pooled_errors() #names=["victor"])
    # analyse_erp_by_subject()

