"""
Terminal report for the offline analysis: print_recording_summary() prints
the per-recording sections (f0, frequency/phase tracking, stimulation, latency).
"""

import os

import numpy as np

from .estimation_errors import _circ_stats, ONLINE_VS_GT, ONLINE_VS_HILBERT, HILBERT_VS_GT, \
    STIM_ONSET_GT, STIM_OFFSET_GT, STIM_ONSET_HILBERT, STIM_OFFSET_HILBERT, \
    STIM_ONSET_ONLINE, STIM_OFFSET_ONLINE, F0_ONLINE_VS_HILBERT

# Terminal names for the LaTeX labels (error series and conditions)
TERMINAL_LABELS = {
    ONLINE_VS_GT:             "online vs GT",
    ONLINE_VS_HILBERT:        "online vs Hilbert",
    HILBERT_VS_GT:            "Hilbert vs GT",
    STIM_ONSET_GT:            "stim onset vs GT",
    STIM_OFFSET_GT:           "stim offset vs GT",
    STIM_ONSET_HILBERT:       "stim onset vs Hilbert",
    STIM_OFFSET_HILBERT:      "stim offset vs Hilbert",
    STIM_ONSET_ONLINE:        "stim onset vs online",
    STIM_OFFSET_ONLINE:       "stim offset vs online",
    r"$\mathrm{P}_1$-Peak":   "P1-Peak",
    r"$\mathrm{P}_1$-Trough": "P1-Trough",
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


def _source_name(recording, graph) -> str:
    if "SimulatedSource" in graph.processors:
        return "SimulatedSource"
    if recording.ground_truth.get("true_phase") is not None:
        return "UDPSource (simulated signal)"
    return "UDPSource"


def print_data_line(graph, recording) -> None:
    fs         = graph.fs
    duration_s = len(recording.raw) / fs
    _row("Data", f"{_source_name(recording, graph)}, {duration_s:.1f} s at {fs:g} Hz"
                 f"{' (from cache)' if recording.from_cache else ''}")


def print_recording_summary(analysis) -> None:
    """f0, frequency tracking, phase tracking, stimulation and latency sections."""
    errors_by_label = {e["label"]: e for e in analysis.errors or []}
    _print_tracking(analysis, errors_by_label)
    if "StimControl" in analysis.graph.processors:
        _print_stimulation(analysis, errors_by_label)
    _print_latencies(analysis)


def _print_tracking(analysis, errors_by_label: dict) -> None:
    """f0, frequency tracking and phase tracking sections."""
    gt      = analysis.recording.ground_truth
    samples = analysis.recording.samples
    offline = analysis.offline

    # f0
    windows = offline.f0_windows
    if offline.f0_note:
        _row("f0", f"{offline.f0:.2f} Hz ({offline.f0_note})")
    elif windows is not None:
        snr = (f", SNR {10 * np.log10(offline.snr):.1f} dB"
               if offline.snr is not None and np.isfinite(offline.snr) and offline.snr > 0 else "")
        if len(windows) > 1:
            spread = f"  (range {np.nanmin(windows):.2f}-{np.nanmax(windows):.2f} over {len(windows)} windows)"
        else:
            spread = ""
        _row("f0 (offline, 20 s windows)", f"{offline.f0:.2f} Hz{spread}{snr}")

    # Frequency tracking
    online_f0 = samples.get("FrequencyEstimation", {}).get("y")
    true_f0   = gt.get("true_inst_freq")
    if online_f0 is not None or true_f0 is not None:
        _table_header("Frequency tracking (Hz)", ["mean", "SD", "MAE"])
        if true_f0 is not None:
            if online_f0 is not None:
                _linear_row("online vs GT", online_f0 - true_f0)
            _linear_row("Hilbert vs GT", offline.f0_continuous - true_f0)
        elif F0_ONLINE_VS_HILBERT in errors_by_label:
            _linear_row("online vs Hilbert", errors_by_label[F0_ONLINE_VS_HILBERT]["values"])

    # Phase tracking
    tracking = [lbl for lbl in (ONLINE_VS_GT, ONLINE_VS_HILBERT, HILBERT_VS_GT)
                if lbl in errors_by_label]
    if tracking:
        _table_header("Phase tracking (deg)", ["mean", "circ. SD", "PLV"])
        for lbl in tracking:
            _circ_row(TERMINAL_LABELS[lbl], errors_by_label[lbl]["values"])


def _print_stimulation(analysis, errors_by_label: dict) -> None:
    """Stimulus-edge phase errors and duration error."""
    from .stimulus import StimulusConfig

    cfg = StimulusConfig.from_graph(analysis.graph)
    duration = (f"{cfg.stim_dur_deg:g} deg" if cfg.stim_dur_unit == "deg" else f"{cfg.stim_dur_ms:g} ms")
    _table_header(f"Stimulation (target onset {cfg.stim_onset_deg:g} deg, duration {duration})",
                  ["mean", "circ. SD", "PLV"])

    for lbl in (STIM_ONSET_GT, STIM_OFFSET_GT, STIM_ONSET_HILBERT, STIM_OFFSET_HILBERT,
                STIM_ONSET_ONLINE, STIM_OFFSET_ONLINE):
        if lbl in errors_by_label:
            _circ_row(TERMINAL_LABELS[lbl], errors_by_label[lbl]["values"])

    dur = errors_by_label.get("Stim duration error")
    if dur is not None:
        _row("duration error", f"{np.nanmean(dur['values']):+.2f} +- {np.nanstd(dur['values']):.2f} ms")


def _print_latencies(analysis) -> None:
    """Pipeline-stage and audio onset latency tables, and UDPSource's true sample rate."""
    latencies = analysis.latency
    if latencies is None:
        print("Not enough processors with source_ts for latency analysis.")
        return

    label_w = 44
    def print_row(label, series):
        """One table row, with when the worst case happened (seconds since the first sample)."""
        lat, ts = series.values, series.ts
        idx = int(np.argmax(lat))
        print(f"  {label:<{label_w}}{np.mean(lat):>9.1f}{np.median(lat):>9.1f}{np.std(lat):>9.1f}"
              f"{np.max(lat):>9.1f}{(ts[idx] - latencies.t0_us) / 1e6:>10.1f} s")

    fs = analysis.graph.fs
    title = f"Pipeline latency (us, sample period {1e6 / fs:g} us)" if fs else "Pipeline latency (us)"
    print(f"\n{title:<{label_w + 2}}{'mean':>9}{'median':>9}{'SD':>9}{'max':>9}{'max at':>12}")
    for stage in latencies.stages:
        print_row(f"{stage.source} -> {stage.target}", stage)
    if latencies.total is not None:
        print_row(f"total {latencies.total.source} -> {latencies.total.target}", latencies.total)

    for note in latencies.audio_notes:
        print(note)
    if latencies.audio is not None:
        print(f"\n{'Audio onset latency (ms)':<{label_w + 2}}"
              f"{'mean':>9}{'median':>9}{'SD':>9}{'max':>9}{'max at':>12}")
        print_row(f"{latencies.audio.source} -> {latencies.audio.target}", latencies.audio)

    if latencies.udp_true_fs is not None:
        if latencies.hw_source is None:
            print("  No true hardware_ts available (pre-fix cache); "
                  "skipping UDPSource hardware->source latency plot.")
        # source_rel, hw_rel = latencies.hw_source
        # dev = hw_rel - source_rel
        # worst = int(np.argmax(np.abs(dev)))
        # print(f"\n  {'UDPSource hardware -> source':<{label_w}}SD {np.std(dev):.1f} us, "
        #       f"max deviation {np.max(np.abs(dev)):.1f} us at {source_rel[worst] / 1e6:.1f} s")

        true_fs    = latencies.udp_true_fs
        nominal_fs = analysis.graph.options("UDPSource").get("fs")
        if nominal_fs:
            drift_ppm = (true_fs - nominal_fs) / nominal_fs * 1e6
            print(f"  {'UDPSource true sample rate':<{label_w}}{true_fs:.4f} Hz "
                  f"(nominal {nominal_fs:g} Hz, drift {drift_ppm:+.2f} ppm)")
        else:
            print(f"  {'UDPSource true sample rate':<{label_w}}{true_fs:.4f} Hz")
