"""
Terminal report for the offline analysis: print_recording_summary() prints
the per-recording sections (data, f0, frequency/phase tracking, stimulation).
"""

import os

import numpy as np

from .core import ONLINE_VS_TRUE, ONLINE_VS_HILBERT, HILBERT_VS_TRUE, \
    STIM_ONSET_HT, STIM_OFFSET_HT, STIM_ONSET_HAT, STIM_OFFSET_HAT

# Terminal names for the (LaTeX) error-series labels from core.compute_errors()
_PHASE_TRACKING_LABELS = {
    ONLINE_VS_TRUE:    "online vs true",
    ONLINE_VS_HILBERT: "online vs offline (Hilbert)",
    HILBERT_VS_TRUE:   "offline (Hilbert) vs true",
}
_STIM_EDGE_LABELS = {
    STIM_ONSET_HT:   "onset vs {ref}",
    STIM_OFFSET_HT:  "offset vs {ref}",
    STIM_ONSET_HAT:  "onset vs online phase",
    STIM_OFFSET_HAT: "offset vs online phase",
}

LABEL_W = 34  # width of the row-label column in every section


def print_header(results_dir: str) -> None:
    name = os.path.basename(os.path.realpath(results_dir.rstrip("/")))
    print(f"\n==== Analysis: {name} ====")


def _row(label: str, text: str) -> None:
    print(f"  {label:<{LABEL_W}}{text}")


def _table_header(title: str, columns: "list[str]") -> None:
    cols = "".join(f"{c:>10}" for c in columns)
    if len(title) > LABEL_W:
        # long title on its own line, so the column names stay above their values
        print(f"\n{title}\n{'':<{LABEL_W + 2}}{cols}")
    else:
        print(f"\n{title:<{LABEL_W + 2}}{cols}")


def _circ_row(label: str, values_deg) -> None:
    from .plot import _circ_stats  # lazy: plot.py imports this module
    vals = np.asarray(values_deg, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        _row(label, f"{'--':>10}{'--':>10}{'--':>10}")
        return
    mu, sd, plv, _ = _circ_stats(np.radians(vals))
    _row(label, f"{np.degrees(mu):>+10.1f}{np.degrees(sd):>10.1f}{plv:>10.2f}")


def _linear_row(label: str, diff) -> None:
    diff = np.asarray(diff, dtype=float)
    _row(label, f"{np.nanmean(diff):>+10.3f}{np.nanstd(diff):>10.3f}{np.nanmean(np.abs(diff)):>10.3f}")


def _source_name(samples: dict, ground_truth: dict) -> str:
    if any(k.startswith("SimulatedSource") for k in samples):
        return "SimulatedSource"
    if ground_truth.get("true_phase") is not None:
        return "UDPSource (simulated signal)"
    return "UDPSource"


def print_data_line(analysis, from_cache: bool) -> None:
    duration_s = len(analysis.ground_truth["raw"]) / analysis.fs
    source = _source_name(analysis.samples, analysis.ground_truth)
    _row("Data", f"{source}, {duration_s:.1f} s at {analysis.fs:g} Hz"
                 f"{' (from cache)' if from_cache else ''}")


def print_recording_summary(analysis) -> None:
    """f0, frequency tracking, phase tracking and stimulation sections."""
    from .stimulus import StimulusConfig

    gt      = analysis.ground_truth
    samples = analysis.samples

    # f0
    windows = analysis.f0_windows
    if analysis.f0_note:
        _row("f0", f"{analysis.f0:.2f} Hz ({analysis.f0_note})")
    elif windows is not None:
        snr = (f", SNR {10 * np.log10(analysis.snr):.1f} dB"
               if analysis.snr is not None and np.isfinite(analysis.snr) and analysis.snr > 0 else "")
        if len(windows) > 1:
            spread = f"  (range {np.nanmin(windows):.2f}-{np.nanmax(windows):.2f} over {len(windows)} windows)"
        else:
            spread = ""
        _row("f0 (offline, 20 s windows)", f"{analysis.f0:.2f} Hz{spread}{snr}")

    # Frequency tracking
    online_f0 = samples.get("FrequencyEstimation", {}).get("y")
    true_f0   = gt.get("true_inst_freq")
    if online_f0 is not None or true_f0 is not None:
        _table_header("Frequency tracking (Hz)", ["mean", "SD", "MAE"])
        if true_f0 is not None:
            if online_f0 is not None:
                _linear_row("online vs true", online_f0 - true_f0)
            _linear_row("offline (Hilbert) vs true", analysis.f0_continuous - true_f0)
        else:
            _linear_row("online vs offline (Hilbert)", online_f0 - analysis.f0_continuous)

    errors_by_label = {e["label"]: e for e in analysis.errors or []}

    # Phase tracking
    tracking = [(name, errors_by_label[lbl]) for lbl, name in _PHASE_TRACKING_LABELS.items()
                if lbl in errors_by_label]
    if tracking:
        _table_header("Phase tracking (deg)", ["mean", "circ. SD", "PLV"])
        for name, err in tracking:
            _circ_row(name, err["values"])

    # Stimulation
    if "StimControl" not in samples:
        return
    cfg = StimulusConfig.from_yaml(analysis.graph_config)
    duration = (f"{cfg.stim_dur_deg:g} deg" if cfg.stim_dur_unit == "deg" else f"{cfg.stim_dur_ms:g} ms")
    _table_header(f"Stimulation (target onset {cfg.stim_onset_deg:g} deg, duration {duration})",
                  ["mean", "circ. SD", "PLV"])
    # counts = analysis.stim_counts
    # if counts is not None:
    #     n_actual, n_expected = counts["delivered"], counts["expected"]

    ref = "true phase" if gt.get("true_phase") is not None else "offline Hilbert phase"
    for lbl, name in _STIM_EDGE_LABELS.items():
        if lbl in errors_by_label:
            _circ_row(name.format(ref=ref), errors_by_label[lbl]["values"])

    dur = errors_by_label.get("Stim duration error")
    if dur is not None:
        _row("duration error", f"{np.nanmean(dur['values']):+.2f} +- {np.nanstd(dur['values']):.2f} ms")
