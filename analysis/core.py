"""
Signal analysis: Hilbert reference and error computation.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import ellip, butter, firwin, filtfilt, sosfiltfilt, hilbert, windows

from .stimulus import StimulusConfig, compute_reference_stimulus, compute_stimulus_edge_errors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PHASE_estimation.jade import jade_v3


def compute_hilbert_reference(
    raw: np.ndarray, fs: float, f0: float, aperiodic_params: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Bandpass + Hilbert transform to produce an offline phase reference.

    When aperiodic_params is given, the spectrum is whitened (divided by the
    fitted 1/f aperiodic PSD) before bandpassing, so the alpha peak is
    isolated from the pink-noise floor rather than just band-limited to it --
    this is deliberate: Hilbert phase is sensitive to off-peak spectral
    content leaking into the passband, so suppressing the aperiodic floor
    gives a cleaner phase estimate. The resulting amplitude is on a
    "whitened" scale, not raw's physical units, but that doesn't matter here
    since only the Hilbert phase (angle) is used downstream, not amplitude.
    """
    X_white = None
    n = len(raw)
    taper = windows.tukey(n, alpha=0.01)
    raw = raw * taper
    if (aperiodic_params is not None) and (all(np.isfinite(aperiodic_params))):
        # Use aperiodic parameters to adjust the bandpass filter
        slope, intercept = aperiodic_params
        freqs = np.fft.rfftfreq(n, d=1/fs)

        X = np.fft.rfft(raw)

        # Aperiodic params were fit to a density-scaled PSD (Welch/periodogram),
        # where Pxx(f) = 2*|X(f)|^2 / (fs*n). Rescale the intercept to match
        # the raw FFT's power units before taking the sqrt to whiten.
        intercept_fft = intercept + np.log10(fs * n / 2)
        L = np.zeros_like(freqs)
        L[1:] = (freqs[1:] ** slope) * (10**intercept_fft)  # skip f=0

        X_white = np.ones_like(X)
        X_white[1:] = X[1:] / np.maximum(1e-12,np.sqrt( np.array(L[1:])))

        raw = np.fft.irfft(X_white, n=n)

    sos      = butter(4, [f0-3, f0+3], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, raw)

    return filtered, np.angle(hilbert(filtered)), X_white


def compute_jade_errors(
    filtered: np.ndarray, time_us: np.ndarray, start_ts: float, fs: float,
    true_phase: np.ndarray, true_inst_freq: np.ndarray,
) -> list[dict]:
    """JADE-based phase/IF estimate (see PHASE_estimation/jade.py) and its
    error against ground truth, in the same format as compute_errors().

    `filtered` must already be bandpass-filtered (e.g. compute_hilbert_reference's
    output). JADE's zero-crossing detector is a bare sign-change test with no
    noise rejection of its own; feeding it the raw signal lets sample-to-sample
    noise (comparable in size to the true per-sample step near a zero-crossing
    when fs >> f0) inject spurious crossings and corrupt the DTW cycle fits.
    """
    t = np.arange(len(filtered)) / fs
    IF, phase, zc, amp = jade_v3(filtered, t, 1 / fs, smooth=0, normalize=0)

    # JADE's phase is zero-crossing-referenced (x ~ sign*amp*sin(2*pi*phase));
    # true_phase is peak-referenced (x ~ amp*cos(phase)). Convert and resolve
    # the sign from JADE's own reconstruction, same as in PHASE_estimation/phase_test.py.
    start = int(zc[0]) - 1
    recon = amp * np.sin(2 * np.pi * phase)
    sign  = 1.0 if np.dot(recon, filtered[start:start + recon.size]) >= 0 else -1.0
    est_phase = 2 * np.pi * phase - sign * np.pi / 2

    n   = len(est_phase)
    t_s = (time_us[start:start + n] - start_ts) / 1e6
    phase_err = np.angle(np.exp(1j * (true_phase[start:start + n] - est_phase)), deg=True)
    if_err    = IF[:n - 1] - true_inst_freq[start:start + n - 1]

    return [
        {"label": "JADE phase error", "time_s": t_s, "values": phase_err, "unit": "degrees"},
        {"label": "JADE IF error", "time_s": t_s[:-1], "values": if_err, "unit": "Hz"},
    ]


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

            # Online estimate vs the offline Hilbert estimate, as if Hilbert
            # were ground truth. When there's no true_phase this is identical
            # to "Phase error", so it's only added here to avoid a duplicate.
            oh_err = np.angle(np.exp(1j * (hilbert_phase[:n] - phi)), deg=True)
            errors.append({"label": "Online vs Hilbert", "time_s": t, "values": oh_err,
                           "unit": "degrees", "linestyle": ":"})

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
