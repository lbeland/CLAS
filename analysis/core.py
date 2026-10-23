"""
Signal analysis: Hilbert reference and error computation.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, windows

from .stimulus import StimulusConfig, compute_reference_stimulus, compute_stimulus_edge_errors, \
    compute_stimulus_duration_error, get_edges

# LaTeX labels of the phase-error series from compute_errors(). They double as
# plot titles/table headers and as the keys other modules look a series up by,
# so import them from here instead of re-typing the strings. Notation:
# theta = true phase (GT), hat theta = online estimate (ONLINE),
# theta_HT = offline Hilbert phase (HILBERT).
# Phase tracking error, <ESTIMATE>_VS_<REFERENCE> = estimate - reference
ONLINE_VS_GT        = r"$\hat\theta - \theta$"
ONLINE_VS_HILBERT   = r"$\hat\theta - \theta_{\mathrm{HT}}$"
HILBERT_VS_GT       = r"$\theta_{\mathrm{HT}} - \theta$"
# Phase error at the stimulus edges, STIM_<EDGE>_<REFERENCE> = reference phase
# at the trigger edge - target phase
STIM_ONSET_GT        = r"Stim onset ($\theta$)"
STIM_OFFSET_GT       = r"Stim offset ($\theta$)"
STIM_ONSET_HILBERT   = r"Stim onset ($\theta_{\mathrm{HT}}$)"
STIM_OFFSET_HILBERT  = r"Stim offset ($\theta_{\mathrm{HT}}$)"
STIM_ONSET_ONLINE    = r"Stim onset ($\hat\theta$)"
STIM_OFFSET_ONLINE   = r"Stim offset ($\hat\theta$)"


def _edge_trim_window(errors: list[dict], lo_frac: float = 0.1,
                      hi_frac: float = 0.90) -> tuple[float, float]:
    """The [t_lo, t_hi] wall-clock window that drops the leading/trailing
    edge-transient slice of a recording.

    Derived from errors[0]'s timestamps at lo_frac/hi_frac of its length, so
    every series is trimmed by the same amount of time regardless of its own
    sampling rate. Used by plot_errors(), the "phase_error" stack panel, and
    estimate_f0_with_phase(), so the trim shown always matches the trim applied.

    Returns:
        (-inf, +inf) when there is nothing to trim on.
    """
    if not errors:
        return (-np.inf, np.inf)
    ref_t = np.asarray(errors[0]["time_s"], dtype=float)
    if ref_t.size == 0:
        return (-np.inf, np.inf)
    lo_idx = int(lo_frac * len(ref_t))
    hi_idx = min(int(hi_frac * len(ref_t)), len(ref_t) - 1)
    return float(ref_t[lo_idx]), float(ref_t[hi_idx])


def compute_hilbert_reference(
    raw: np.ndarray, fs: float, f0: float, aperiodic_params: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Bandpass + Hilbert transform to produce an offline phase reference.

    Args:
        raw: Input signal.
        fs: Sample rate (Hz).
        f0: Bandpass centre frequency (Hz); passband is [0.7, 1.3] * f0.
        aperiodic_params: (slope, intercept) of a fitted 1/f PSD. When given,
            the spectrum is whitened before bandpassing so the alpha peak is
            isolated from the pink-noise floor, giving a cleaner Hilbert
            phase (amplitude ends up on a whitened scale, which is fine since
            only the phase is used downstream).

    Returns:
        (filtered signal, Hilbert phase, whitened spectrum or None).
    """
    X_white = None

    if (aperiodic_params is not None) and (all(np.isfinite(aperiodic_params))):
        n = len(raw)
        taper = windows.tukey(n, alpha=0.01)
        raw = raw * taper
        # Use aperiodic parameters to adjust the bandpass filter
        slope, intercept = aperiodic_params
        freqs = np.fft.rfftfreq(n, d=1/fs)

        X = np.fft.rfft(raw)

        # Rescale intercept from density-scaled PSD units (Pxx = 2*|X|^2/(fs*n)) to raw FFT power
        intercept_fft = intercept + np.log10(fs * n / 2)
        L = np.zeros_like(freqs)
        L[1:] = (freqs[1:] ** slope) * (10**intercept_fft)  # skip f=0

        X_white = np.ones_like(X)
        X_white[1:] = X[1:] / np.maximum(1e-12,np.sqrt( np.array(L[1:])))
        X_white[0] = 1.0  # avoid a DC offset in the whitened signal

        raw = np.fft.irfft(X_white, n=n)

    sos      = butter(4, [0.7*f0, 1.3*f0], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, raw)

    return filtered, np.angle(hilbert(filtered)), X_white

def compute_errors(
    samples: dict,
    ground_truth: dict,
    hilbert_phase: np.ndarray,
    start_ts: float,
    fs: float,
    graph_config: dict,
) -> tuple[list[dict], np.ndarray | None, dict | None]:
    """
    Compute phase, f0, and stimulus edge errors.
    Returns (list of error dicts, stim_ref array or None, stimulus onset
    counts {"delivered", "expected"} or None).

    Sign convention for every error series here: estimate - reference
    (wrapped to (-180, 180] for phases). Positive => the estimate leads /
    overshoots the reference.
    """
    errors         = []
    true_phase     = ground_truth["true_phase"]
    true_inst_freq = ground_truth["true_inst_freq"]

    # Phase error
    if samples.get("PhaseEstimation_phase") is not None:
        t   = (samples["PhaseEstimation_phase"]["x"] - start_ts) / 1e6
        phi = samples["PhaseEstimation_phase"]["y"]
        n   = len(phi)
        ref = true_phase[:n] if true_phase is not None else hilbert_phase[:n]
        label = ONLINE_VS_GT if true_phase is not None else ONLINE_VS_HILBERT
        err = np.angle(np.exp(1j * (phi - ref)), deg=True)
        errors.append({"label": label, "time_s": t, "values": err,
                        "unit": "degrees", "linestyle": "-"})

        if true_phase is not None:
            h_err = np.angle(np.exp(1j * (hilbert_phase[:n] - true_phase[:n])), deg=True)
            errors.append({"label": HILBERT_VS_GT, "time_s": t, "values": h_err,
                           "unit": "degrees", "linestyle": "-"})

            # Online vs offline Hilbert estimate
            oh_err = np.angle(np.exp(1j * (phi - hilbert_phase[:n])), deg=True)
            errors.append({"label": ONLINE_VS_HILBERT, "time_s": t, "values": oh_err,
                           "unit": "degrees", "linestyle": "--"})

    # f0 error
    if samples.get("FrequencyEstimation") is not None and true_inst_freq is not None:
        t   = (samples["FrequencyEstimation"]["x"] - start_ts) / 1e6
        f0 = samples["FrequencyEstimation"]["y"]
        n   = min(len(f0), len(true_inst_freq))
        errors.append({"label": "f0 error", "time_s": t[:n],
                       "values": f0[:n] - true_inst_freq[:n], "unit": "Hz"})

    # Stimulus edge errors
    stim_ref = None
    stim_counts = None
    if samples.get("StimControl") is not None:
        trigger_y = (
            samples["UDPSource_TRIGGER"]["y"]
            if "UDPSource_TRIGGER" in samples
            else samples["StimControl"]["y"]
        )
        ref_phase = true_phase if true_phase is not None else hilbert_phase
        onset_label, offset_label = ((STIM_ONSET_GT, STIM_OFFSET_GT) if true_phase is not None
                                     else (STIM_ONSET_HILBERT, STIM_OFFSET_HILBERT))

        if true_inst_freq is not None:
            f0_y = true_inst_freq
        elif samples.get("FrequencyEstimation") is not None:
            f0_y = samples["FrequencyEstimation"]["y"]
        else:
            print("No f0 information available; using constant 10 Hz for stimulus reconstruction.")
            f0_y = np.full(len(ground_truth["time"]), 10.0)

        stim_cfg       = StimulusConfig.from_yaml(graph_config)
        trigger_binary = (trigger_y > 0.5).astype(float)
        time_us        = ground_truth["time"]

        # Ideal stimulus train, for visualisation / EDF export
        stim_ref, _, _ = compute_reference_stimulus(ref_phase, f0_y, fs, stim_cfg)
        stim_counts = {"delivered": len(get_edges(trigger_binary, "rising")),
                       "expected":  len(get_edges(stim_ref, "rising"))}

        # Phase error at each trigger edge vs the controller's target phase,
        # scored against true_phase if available, else the offline Hilbert phase
        onset_err, offset_err = compute_stimulus_edge_errors(
            trigger_binary, time_us, ref_phase, stim_cfg, start_ts, fs)
        if onset_err is not None:
            errors.append({"label": onset_label,
                           "time_s": onset_err["time_s"], "values": onset_err["values"],
                           "unit": "degrees"})
        if offset_err is not None:
            errors.append({"label": offset_label,
                           "time_s": offset_err["time_s"], "values": offset_err["values"],
                           "unit": "degrees", "linestyle": "--"})

        # "ms" mode times the burst rather than phase-targeting the falling edge,
        # so score actual-vs-target burst length in ms instead
        if stim_cfg.stim_dur_unit != "deg":
            dur_err = compute_stimulus_duration_error(trigger_binary, time_us, stim_cfg, start_ts)
            if dur_err is not None:
                errors.append({"label": "Stim duration error",
                               "time_s": dur_err["time_s"], "values": dur_err["values"],
                               "unit": "ms"})

        # Same edges, scored against the live ECHT PhaseEstimation output instead
        echt_phase = samples.get("PhaseEstimation_phase", {}).get("y")
        echt_onset_err, echt_offset_err = compute_stimulus_edge_errors(
            trigger_binary, time_us, echt_phase, stim_cfg, start_ts, fs)
        if echt_onset_err is not None:
            errors.append({"label": STIM_ONSET_ONLINE,
                            "time_s": echt_onset_err["time_s"], "values": echt_onset_err["values"],
                            "unit": "degrees", "linestyle": ":"})
        if echt_offset_err is not None:
            errors.append({"label": STIM_OFFSET_ONLINE,
                            "time_s": echt_offset_err["time_s"], "values": echt_offset_err["values"],
                            "unit": "degrees", "linestyle": ":"})

    return errors, stim_ref, stim_counts
