"""
Pooled error analysis: phase/f0/stimulus-edge errors of every recording in the
session folders, combined into polar histograms, a per-recording scatter, a
per-condition error table and pooled mean +/- SD metrics.
    python -m analysis.main pooled   (from CLAS/)
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from .clas_file_reader import find_session_run_dirs
from .estimation_errors import _circ_stats, ONLINE_VS_GT, ONLINE_VS_HILBERT, HILBERT_VS_GT, \
    STIM_ONSET_GT, STIM_OFFSET_GT, STIM_ONSET_HILBERT, STIM_OFFSET_HILBERT, \
    STIM_ONSET_ONLINE, STIM_OFFSET_ONLINE, F0_ONLINE_VS_GT, F0_ONLINE_VS_HILBERT
from .pipeline import RecordingAnalysis, iter_analyses
from .plot import plot_errors, plot_pooled_scatter, write_condition_error_table
from .stimulus import STIM_ONSET_DEGS, CONDITION_LABELS

# Column order for the per-condition error table
_CONDITION_TABLE_COLUMNS = [
    ONLINE_VS_GT, ONLINE_VS_HILBERT, HILBERT_VS_GT,
    STIM_ONSET_GT, STIM_OFFSET_GT, STIM_ONSET_HILBERT, STIM_OFFSET_HILBERT,
    STIM_ONSET_ONLINE, STIM_OFFSET_ONLINE,
]

# Row order for the per-condition error table: the four phase-offset
# conditions first, then a rule, then the two P1 (peak/trough) conditions.
_CONDITION_ROW_GROUPS = [[60, 150, 240, 330], [0, 180]]

# Non-circular metrics printed as a single pooled mean +/- SD, not split by condition
_GLOBAL_METRIC_LABELS = (F0_ONLINE_VS_GT, "Stim duration error", F0_ONLINE_VS_HILBERT, "Total latency")


def analyse_pooled_errors(f0: float = 10, names: list[str] = None, base_dir: str = "results",
                          session_glob: str = "CLAS_*") -> None:
    """Pool phase/f0/stimulus-edge errors across every recording and plot/print results.

    Combines every recording under every session folder matching
    session_glob (e.g. results/CLAS_*/<run>) and:
      - Plots pooled polar error-distribution histograms, as
        analyse_results() does per-recording, but combined across all runs.
        No time-window trim is applied: Hilbert-derived series are already
        NaN-masked at their own edges (see estimate_offline()).
      - Plots per-recording scatter (plot_pooled_scatter) of each
        recording's circular mean +/- SD phase error against its online
        f0-estimate variance and offline alpha-peak SNR.
      - Prints/writes a table of circular mean +/- SD per degree-unit error
        label, one row per stim_onset_deg condition (see
        stimulus.STIM_ONSET_DEGS/CONDITION_LABELS) and one column per label.
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
        run_dirs = [d for d in run_dirs if any(name.lower() in d.lower() for name in names)]
    print(f"Found {len(run_dirs)} recording(s) under {base_dir}/{session_glob}/*.")

    pooled: dict[str, dict] = {}            # label -> series metadata + one values array per recording
    by_condition: dict[tuple, list] = {}    # (stim_onset_deg, label) -> one degree array per recording
    scatter_records: list[dict] = []
    for results_dir, analysis in iter_analyses(run_dirs, f0):
        condition = analysis.graph.options("StimControl").get("stim_onset_deg")
        for err in _pooling_series(analysis):
            vals = np.asarray(err["values"], dtype=float)
            if err["unit"] == "degrees":
                vals = vals[~np.isnan(vals)]
                if condition in STIM_ONSET_DEGS and vals.size:
                    by_condition.setdefault((condition, err["label"]), []).append(vals)
            entry = pooled.setdefault(err["label"], {
                "label": err["label"], "unit": err["unit"], "linestyle": err.get("linestyle", "-"),
                "values": [],
            })
            entry["values"].append(vals)

        record = _scatter_record(os.path.basename(results_dir), analysis)
        if record is not None:
            scatter_records.append(record)

    if not pooled:
        print("No errors collected; nothing to plot.")
        return

    merged_errors = [{"label": e["label"], "unit": e["unit"], "linestyle": e["linestyle"],
                      "values": np.concatenate(e["values"])}
                     for e in pooled.values()]
    plot_errors(merged_errors)
    plot_pooled_scatter(scatter_records)
    _write_condition_table(by_condition)
    _print_global_metrics(pooled)
    plt.show()


def _pooling_series(analysis: RecordingAnalysis) -> list[dict]:
    """The recording's error series, plus its end-to-end pipeline latency
    (the same total the single-recording report prints)."""
    series = list(analysis.errors)
    total = analysis.latency.total if analysis.latency is not None else None
    if total is not None:
        time = analysis.recording.time
        n = len(total.values)
        series.append({"label": "Total latency", "unit": "us",
                       "time_s": (time[:n] - time[0]) / 1e6,
                       "values": total.values.astype(float)})
    return series


def _scatter_record(name: str, analysis: RecordingAnalysis) -> "dict | None":
    """One plot_pooled_scatter() bullet per metric: the recording's online vs
    Hilbert phase error and stimulus-onset errors, against its offline SNR.
    None without an online vs Hilbert phase error series."""
    errors_by_label = {e["label"]: e for e in analysis.errors}
    if ONLINE_VS_HILBERT not in errors_by_label:
        return None

    def values(label):
        err = errors_by_label.get(label)
        return np.asarray(err["values"], dtype=float) if err is not None else np.array([])

    snr = analysis.offline.snr
    return {
        "name":             name,
        "phase_vals":       values(ONLINE_VS_HILBERT),
        STIM_ONSET_ONLINE:  values(STIM_ONSET_ONLINE),
        STIM_ONSET_HILBERT: values(STIM_ONSET_HILBERT),
        "snr_db":           10.0 * np.log10(snr) if snr is not None and np.isfinite(snr) and snr > 0 else np.nan,
    }


def _write_condition_table(by_condition: dict[tuple, list]) -> None:
    """Circular mean +/- SD and PLV per (stim_onset_deg condition, error label)."""
    if not by_condition:
        print("No recording matched a known stim_onset_deg condition; skipping condition table.")
        return

    condition_stats: dict[int, dict[str, tuple[float, float, float]]] = {}
    for (condition, label), chunks in by_condition.items():
        mu, sd, plv, _ = _circ_stats(np.radians(np.concatenate(chunks)))
        condition_stats.setdefault(condition, {})[label] = (np.degrees(mu), np.degrees(sd), plv)

    labels  = list(dict.fromkeys(label for _, label in by_condition))  # first-seen order
    columns = [c for c in _CONDITION_TABLE_COLUMNS if c in labels] + \
              sorted(c for c in labels if c not in _CONDITION_TABLE_COLUMNS)
    write_condition_error_table(condition_stats, columns, _CONDITION_ROW_GROUPS, CONDITION_LABELS)


def _print_global_metrics(pooled: dict[str, dict]) -> None:
    """Plain (non-circular) pooled mean +/- SD of the _GLOBAL_METRIC_LABELS series."""
    for label, entry in pooled.items():
        if label not in _GLOBAL_METRIC_LABELS or entry["unit"] == "degrees":
            continue
        chunks = [v[~np.isnan(v)] for v in entry["values"]]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            continue
        vals = np.concatenate(chunks)
        unit = entry["unit"]
        print(f"{label}: {np.mean(vals):.3f} +- {np.std(vals):.3f} {unit} "
              f"(n={vals.size} samples, {len(chunks)} recording(s))")
        if label == F0_ONLINE_VS_HILBERT:
            print(f"Online vs offline MAE: {np.mean(np.abs(vals)):.3f} {unit}")
        if label == "Stim duration error":
            print(f"Stim duration MAE: {np.mean(np.abs(vals)):.3f} {unit}")
