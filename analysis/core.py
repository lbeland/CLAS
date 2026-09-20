"""
Signal analysis: Hilbert reference and error computation.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, windows
from tqdm import tqdm

from .stimulus import StimulusConfig, compute_reference_stimulus, compute_stimulus_edge_errors, \
    compute_stimulus_duration_error, get_edges

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PHASE_estimation.jade import jade_v3


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


def _fif_isolate_component(
    raw: np.ndarray, fs: float, target_hz: float,
    alpha: int = 90, ext_points: int = 30,
) -> np.ndarray | None:
    """Decompose `raw` with Fast Iterative Filtering and return the IMC whose
    mean frequency is closest to `target_hz`.

    Adaptive alternative to the linear bandpass used elsewhere for JADE
    conditioning: FIF splits the signal into intrinsic mode components that
    are mono-component by construction, rather than merely band-limited.

    Returns:
        The matching IMC, or None if FIF is unavailable or fails (caller
        falls back to the bandpassed signal).
    """
    try:
        from PHASE_estimation.FIF import FIF as FIFClass
    except Exception as exc:  # numba / package missing
        print(f"FIF unavailable ({exc}); JADE will use the bandpassed signal.")
        return None

    try:
        fif = FIFClass(alpha=alpha, ExtPoints=ext_points)
        fif.run(np.asarray(raw, dtype=float))
        imc = fif.data["IMC"]
        imc_freqs, _ = fif.get_freq_amplitudes(dt=1 / fs, as_output=True)
    except Exception as exc:
        print(f"FIF decomposition failed ({exc}); JADE will use the bandpassed signal.")
        return None

    if imc is None or imc.shape[0] < 2:
        return None

    # Ignore the trend (last row) when matching to the target frequency.
    idx = int(np.argmin(np.abs(imc_freqs[:-1] - target_hz)))
    print(f"FIF: {imc.shape[0]} IMCs at {np.array2string(imc_freqs, precision=2)} Hz; "
          f"JADE on IMC #{idx} (mean {imc_freqs[idx]:.2f} Hz, target {target_hz:.2f} Hz)")
    return imc[idx]


def compute_jade_errors(
    filtered: np.ndarray, time_us: np.ndarray, start_ts: float, fs: float,
    true_phase: np.ndarray, true_inst_freq: np.ndarray,
    raw: np.ndarray | None = None, f0: float | None = None, use_fif: bool = True,
) -> list[dict]:
    """JADE-based phase/IF estimate (see PHASE_estimation/jade.py) and its
    error against ground truth, in the same format as compute_errors().

    JADE's zero-crossing detector has no noise rejection, so it always runs
    on a conditioned, mono-component signal rather than a wideband one:
    Fast Iterative Filtering picks the IMC nearest the target frequency when
    `use_fif` and `raw` are given, else `filtered` is used directly (must
    already be bandpass-filtered, e.g. compute_hilbert_reference's output).
    """
    jade_input = None
    if use_fif and raw is not None:
        target_hz = (
            float(f0) if (f0 is not None and np.isfinite(f0))
            else float(np.nanmean(true_inst_freq))
        )
        jade_input = _fif_isolate_component(raw, fs, target_hz)
    if jade_input is None:
        jade_input = filtered

    t = np.arange(len(jade_input)) / fs
    IF, phase, zc, amp = jade_v3(jade_input, t, 1 / fs, smooth=0, normalize=0)

    # Convert JADE's zero-crossing-referenced phase to true_phase's peak-referenced
    # convention, resolving the sign from JADE's own reconstruction
    start = int(zc[0]) - 1
    recon = amp * np.sin(2 * np.pi * phase)
    sign  = 1.0 if np.dot(recon, jade_input[start:start + recon.size]) >= 0 else -1.0
    est_phase = 2 * np.pi * phase - sign * np.pi / 2

    n   = len(est_phase)
    t_s = (time_us[start:start + n] - start_ts) / 1e6
    # Sign convention: estimate - reference (matches compute_errors()).
    phase_err = np.angle(np.exp(1j * (est_phase - true_phase[start:start + n])), deg=True)
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
    Compute phase, f0, and stimulus edge errors.
    Returns (list of error dicts, stim_ref array or None).

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
        label = r"$\hat \theta - \theta$" if true_phase is not None else r"$\hat\theta - \theta_{\mathrm{HT}}$"
        err = np.angle(np.exp(1j * (phi - ref)), deg=True)
        errors.append({"label": label, "time_s": t, "values": err,
                        "unit": "degrees", "linestyle": "-"})

        if true_phase is not None:
            h_err = np.angle(np.exp(1j * (hilbert_phase[:n] - true_phase[:n])), deg=True)
            errors.append({"label": r"$\theta_{\mathrm{HT}} - \theta$", "time_s": t, "values": h_err,
                           "unit": "degrees", "linestyle": "-"})

            # Online vs offline Hilbert estimate; equals "Phase error" without true_phase
            oh_err = np.angle(np.exp(1j * (phi - hilbert_phase[:n])), deg=True)
            errors.append({"label": r"$\hat\theta - \theta_{\mathrm{HT}}$", "time_s": t, "values": oh_err,
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
    if samples.get("StimControl") is not None:
        trigger_y = (
            samples["UDPSource_TRIGGER"]["y"]
            if "UDPSource_TRIGGER" in samples
            else samples["StimControl"]["y"]
        )
        ref_phase = true_phase if true_phase is not None else hilbert_phase

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
        n_ideal  = len(get_edges(stim_ref, "rising"))
        n_actual = len(get_edges(trigger_binary, "rising"))
        if n_actual != n_ideal:
            print(f"Stimulus: {n_actual} onset(s) delivered vs {n_ideal} ideal.")

        # Phase error at each trigger edge vs the controller's target phase,
        # scored against true_phase if available, else the offline Hilbert phase
        onset_err, offset_err = compute_stimulus_edge_errors(
            trigger_binary, time_us, ref_phase, stim_cfg, start_ts, fs)
        if onset_err is not None:
            errors.append({"label": r"Stim onset ($\theta_{\mathrm{HT}}$)",
                           "time_s": onset_err["time_s"], "values": onset_err["values"],
                           "unit": "degrees"})
        if offset_err is not None:
            errors.append({"label": r"Stim offset ($\theta_{\mathrm{HT}}$)",
                           "time_s": offset_err["time_s"], "values": offset_err["values"],
                           "unit": "degrees", "linestyle": "--"})

        # "ms" mode times the burst rather than phase-targeting the falling edge,
        # so score actual-vs-target burst length in ms instead
        if stim_cfg.stim_dur_unit != "deg":
            dur_err = compute_stimulus_duration_error(trigger_binary, time_us, stim_cfg, start_ts)
            if dur_err is not None:
                print(f"Stim duration error: {np.nanmean(dur_err['values']):.2f} ms "
                      f"+- {np.nanstd(dur_err['values']):.2f} ms")
                errors.append({"label": "Stim duration error",
                               "time_s": dur_err["time_s"], "values": dur_err["values"],
                               "unit": "ms"})

        # Same edges, scored against the live ECHT PhaseEstimation output instead
        echt_phase = samples.get("PhaseEstimation_phase", {}).get("y")
        echt_onset_err, echt_offset_err = compute_stimulus_edge_errors(
            trigger_binary, time_us, echt_phase, stim_cfg, start_ts, fs)
        if echt_onset_err is not None:
            errors.append({"label": r"Stim onset ($\hat\theta$)",
                            "time_s": echt_onset_err["time_s"], "values": echt_onset_err["values"],
                            "unit": "degrees", "linestyle": ":"})
        if echt_offset_err is not None:
            errors.append({"label": r"Stim offset ($\hat\theta$)",
                            "time_s": echt_offset_err["time_s"], "values": echt_offset_err["values"],
                            "unit": "degrees", "linestyle": ":"})

    return errors, stim_ref
