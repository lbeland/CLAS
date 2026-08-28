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

from analysis.loader       import load_processor_signals, analyse_latencies, extract_ground_truth
from analysis.iaf          import estimate_iaf, estimate_iaf_with_phase
from analysis.core         import compute_hilbert_reference, compute_errors, compute_jade_errors
from analysis.edf_io       import write_raw_signals_edf, load_runtime, write_analysis_edf
from analysis.runtime_meta import write_runtime_metadata
from analysis.plot         import plot_errors, plot_spectrum, plot_time_series, \
                                  plot_iaf, get_erp_windows, plot_erp_latency, load_stim_annotations


@dataclass
class RecordingAnalysis:
    """Everything derived from one results folder: the loaded (or cached)
    recording plus the offline IAF/Hilbert/error analysis of it. Shared by
    analyse_results() (which additionally plots/exports it) and
    compute_recording_errors() (which just wants .errors)."""
    graph_file:       str
    graph_config:     dict
    fs:               float
    samples:          dict
    ground_truth:     dict
    annotations:      "mne.Annotations | None"
    f0:               float
    aperiodic_params: np.ndarray
    raw:              np.ndarray
    filtered:         np.ndarray
    hilbert_phase:    np.ndarray
    X_white:          "np.ndarray | None"
    iaf_continuous:   np.ndarray
    errors:           list[dict]
    stim_ref:         "np.ndarray | None"


def load_and_analyse(f0: float, results_dir: str) -> "RecordingAnalysis | None":
    """Load one results folder (from cache if available, else the raw .bin
    files) and run the offline analysis on it: IAF estimate, Hilbert
    reference, phase/IAF/stimulus-edge errors. Returns None if the folder
    can't be loaded (no graph config / no source signal)."""
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
        ground_truth = extract_ground_truth(samples)
        if ground_truth is None:
            print("Error: no source signal found (SourceClient or Producer). Aborting.")
            return None

        # Persist the raw/runtime signals so future runs can skip this branch
        annotations = load_stim_annotations(results_dir, ground_truth["time"][0])
        write_raw_signals_edf(raw_edf_path, fs, samples, ground_truth, annotations=annotations)
        write_runtime_metadata(meta_h5_path, fs, samples, ground_truth)

    # Estimate IAF offline on the full recording
    iaf, aperiodic_params = estimate_iaf(ground_truth["raw"], fs)
    if np.all(~np.isfinite(iaf)):
        print("Could not estimate IAF from ground truth; using default f0 =", f0)
    else:
        f0 = float(np.nanmean(iaf))
        print(f"Estimated IAFs on 30-second windows: mean={f0:.2f} Hz, all={iaf}")

    # Offline Hilbert reference
    raw = ground_truth["raw"]
    filtered, hilbert_phase, X_white = compute_hilbert_reference(raw, fs, f0, aperiodic_params=aperiodic_params)
    iaf_continuous = estimate_iaf_with_phase(hilbert_phase, fs, f0)

    # Compute errors
    start_ts = ground_truth["time"][0]
    errors, stim_ref = compute_errors(samples, ground_truth, hilbert_phase, start_ts, fs, graph_config)

    # Producer ground truth (true_inst_freq known) + short enough for JADE's DTW cost to be feasible
    if ground_truth["true_inst_freq"] is not None and len(raw) / fs <= 20:
        errors += compute_jade_errors(filtered, ground_truth["time"], start_ts, fs,
                                       ground_truth["true_phase"], ground_truth["true_inst_freq"])

    return RecordingAnalysis(
        graph_file=graph_file, graph_config=graph_config, fs=fs,
        samples=samples, ground_truth=ground_truth, annotations=annotations,
        f0=f0, aperiodic_params=aperiodic_params, raw=raw, filtered=filtered,
        hilbert_phase=hilbert_phase, X_white=X_white, iaf_continuous=iaf_continuous,
        errors=errors, stim_ref=stim_ref,
    )


def analyse_results(f0: float, results_dir: str, show: bool = True) -> None:
    analysis = load_and_analyse(f0, results_dir)
    if analysis is None:
        return

    # Pipeline latency analysis. Reproducible from cache too, since
    # runtime_metadata.h5 preserves source_ts exactly (unlike an EDF round-trip).
    analyse_latencies(analysis.samples, analysis.ground_truth, analysis.graph_config)

    # ERPCLAS-specific: ERP average plot
    if os.path.basename(analysis.graph_file) == "ERPCLAS.yaml":
        windows = get_erp_windows(analysis.fs, analysis.samples, channel=[4,3,27,26], apply_csd=False)
        plot_erp_latency(windows, analysis.fs)

    # Export analysis outputs to EDF (cheap; rebuilt every run from the cached runtime signals)
    analysis_path = os.path.join(results_dir, "analysis.edf")
    write_analysis_edf(analysis_path, analysis.fs, analysis.ground_truth, analysis.samples,
                        analysis.filtered, analysis.hilbert_phase, analysis.iaf_continuous,
                        analysis.stim_ref, annotations=analysis.annotations)

    # Plot errors
    plot_errors(analysis.errors)

    # IAF time series
    start_ts = analysis.ground_truth["time"][0]
    plot_iaf(analysis.ground_truth, analysis.iaf_continuous, analysis.samples, start_ts)

    # Plot spectrum
    plot_spectrum(analysis.raw, analysis.samples, analysis.filtered, analysis.fs,
                  analysis.aperiodic_params, analysis.X_white)

    # Plot time series
    time_range = None #(10, 20)  # set to e.g. (17, 18) to zoom in seconds
    # plot_time_series(analysis.ground_truth, analysis.samples, analysis.hilbert_phase,
    #                   analysis.stim_ref, time_range=time_range)

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
            analyse_results(f0, results_dir, show=False)
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


def compute_recording_errors(f0: float, results_dir: str) -> list[dict]:
    """Load one results folder and compute its phase/IAF/stimulus-edge
    errors, without writing files or plotting. Returns [] if the folder
    can't be loaded or has no source signal."""
    analysis = load_and_analyse(f0, results_dir)
    return analysis.errors if analysis is not None else []


def find_session_run_dirs(base_dir: str = "results", session_glob: str = "CLAS_*") -> list[str]:
    """Every TurboLinkCLAS.yaml run directory nested one level under a
    session folder matching session_glob (e.g. results/CLAS_linda/<run>)."""
    session_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, session_glob)) if os.path.isdir(d))
    return sorted({
        os.path.dirname(p)
        for session_dir in session_dirs
        for p in glob.glob(os.path.join(session_dir, "*", "TurboLinkCLAS.yaml"))
    })


def analyse_pooled_errors(f0: float = 10, names: list[str] = None, base_dir: str = "results", session_glob: str = "CLAS_*") -> None:
    """Pool phase/IAF/stimulus-edge errors across every recording nested
    under every session folder matching session_glob (e.g. every
    results/CLAS_*/<run>), and plot them exactly like analyse_results() does
    for a single recording (time series + polar histograms) -- but combined
    across all of them, one polar distribution per error label."""
    run_dirs = find_session_run_dirs(base_dir, session_glob)
    if names is not None:
        run_dirs_selected = [d for d in run_dirs if any(name.lower() in d.lower() for name in names)]
        run_dirs = run_dirs_selected
    print(f"Found {len(run_dirs)} recording(s) under {base_dir}/{session_glob}/*.")

    pooled: dict[str, dict] = {}
    t_offset = 0.0
    for i, results_dir in enumerate(run_dirs, 1):
        print(f"[{i}/{len(run_dirs)}] {results_dir}")
        try:
            errors = compute_recording_errors(f0, results_dir)
        except Exception:
            print(f"  FAILED: {results_dir}")
            traceback.print_exc()
            continue

        # Offset each recording's error times so the pooled time-series plot
        # lays them out sequentially instead of overlapping at t=0; the
        # polar histograms only look at "values", so this doesn't affect them.
        recording_max_t = max((np.max(err["time_s"]) for err in errors if len(err["time_s"])), default=0.0)
        for err in errors:
            entry = pooled.setdefault(err["label"], {
                "label": err["label"], "unit": err["unit"], "linestyle": err.get("linestyle", "-"),
                "time_s": [], "values": [],
            })
            entry["time_s"].append(err["time_s"] + t_offset)
            entry["values"].append(err["values"])
        t_offset += recording_max_t + 1.0

    if not pooled:
        print("No errors collected; nothing to plot.")
        return

    merged_errors = [
        {"label": e["label"], "unit": e["unit"], "linestyle": e["linestyle"],
         "time_s": np.concatenate(e["time_s"]), "values": np.concatenate(e["values"])}
        for e in pooled.values()
    ]

    plot_errors(merged_errors)
    plt.show()


if __name__ == "__main__":
    analyse_all_results(f0=10)
    # analyse_results(f0=10, results_dir="results/simu_4.0_staticf0_20260709_093626")
    # analyse_pooled_errors(f0=10) #, names=["victor", "dorothea"])
