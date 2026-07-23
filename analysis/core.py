"""
Signal analysis: Hilbert reference and error computation.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from .stimulus import StimulusConfig, compute_reference_stimulus, compute_stimulus_edge_errors


def compute_hilbert_reference(
    raw: np.ndarray, fs: float, f0: float, aperiodic_params: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass + Hilbert transform to produce an offline phase reference."""
    if aperiodic_params is not None:
        # Use aperiodic parameters to adjust the bandpass filter
        slope, intercept = aperiodic_params
        n = len(raw)
        freqs = np.fft.rfftfreq(n, d=1/fs)
        X = np.fft.rfft(raw)

        # Aperiodic power model L(f) = f^-chi * 10^b
        L = np.zeros_like(freqs)
        L[1:] = freqs[1:] ** (slope) * 10**intercept  # skip f=0

        X_white = X.copy()
        X_white[1:] = X[1:] / np.sqrt(L[1:])

        raw_white = np.fft.irfft(X_white, n=n)

    sos      = butter(4, [6.0, 16.0], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, raw_white if aperiodic_params is not None else raw)

    filtered = raw_white if aperiodic_params is not None else raw
    return filtered, np.angle(hilbert(filtered))


def compute_errors(
    samples: dict,
    ground_truth: dict,
    hilbert_phase: np.ndarray,
    start_ts: float,
    fs: float,
    graph_config: dict,
) -> tuple[list[dict], np.ndarray | None]:
    """
    Compute phase, IAF, and stimulus edge errors.
    Returns (list of error dicts, stim_ref array or None).
    """
    errors         = []
    true_phase     = ground_truth["true_phase"]
    true_inst_freq = ground_truth["true_inst_freq"]

    # Phase error
    if samples.get("PhaseEstimator_phase") is not None:
        t   = (samples["PhaseEstimator_phase"]["x"] - start_ts) / 1e6
        phi = samples["PhaseEstimator_phase"]["y"]
        n   = len(phi)
        ref = true_phase[:n] if true_phase is not None else hilbert_phase[:n]
        err = np.angle(np.exp(1j * (ref - phi)), deg=True)
        errors.append({"label": "Phase error", "time_s": t, "values": err, "unit": "degrees"})

        if true_phase is not None:
            h_err = np.angle(np.exp(1j * (true_phase[:n] - hilbert_phase[:n])), deg=True)
            errors.append({"label": "Hilbert ref error", "time_s": t, "values": h_err,
                           "unit": "degrees", "linestyle": "--"})

    # IAF error
    if samples.get("IAFEstimator") is not None and true_inst_freq is not None:
        t   = (samples["IAFEstimator"]["x"] - start_ts) / 1e6
        iaf = samples["IAFEstimator"]["y"]
        n   = min(len(iaf), len(true_inst_freq))
        errors.append({"label": "IAF error", "time_s": t[:n],
                       "values": iaf[:n] - true_inst_freq[:n], "unit": "Hz"})

    # Stimulus edge errors
    stim_ref = None
    if samples.get("StimulusController") is not None:
        trigger_y = (
            samples["SourceClient_TRIGGER"]["y"]
            if "SourceClient_TRIGGER" in samples
            else samples["StimulusController"]["y"]
        )
        ref_phase = true_phase if true_phase is not None else hilbert_phase

        if true_inst_freq is not None:
            iaf_y = true_inst_freq
        elif samples.get("IAFEstimator") is not None:
            iaf_y = samples["IAFEstimator"]["y"]
        else:
            print("No IAF information available; using constant 10 Hz for stimulus reconstruction.")
            iaf_y = np.full(len(ground_truth["time"]), 10.0)

        stim_cfg = StimulusConfig.from_yaml(graph_config)
        stim_ref, _, _ = compute_reference_stimulus(ref_phase, iaf_y, fs, stim_cfg)

        trigger_binary = (trigger_y > 0.5).astype(float)
        onset_err, offset_err = compute_stimulus_edge_errors(
            ref_signal=stim_ref,
            actual_signal=trigger_binary,
            time_us=ground_truth["time"],
            phase=ref_phase,
            start_ts=start_ts,
        )
        if onset_err is not None:
            errors.append({"label": "Stim onset error",
                           "time_s": onset_err["time_s"], "values": onset_err["values"],
                           "unit": "degrees"})
        if offset_err is not None:
            errors.append({"label": "Stim offset error",
                           "time_s": offset_err["time_s"], "values": offset_err["values"],
                           "unit": "degrees", "linestyle": "--"})

    return errors, stim_ref
