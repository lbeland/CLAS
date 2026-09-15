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
from .f0 import alpha_fast

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

    if (aperiodic_params is not None) and (all(np.isfinite(aperiodic_params))):
        n = len(raw)
        taper = windows.tukey(n, alpha=0.01)
        raw = raw * taper
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

    This is the adaptive alternative to the linear bandpass used elsewhere for
    JADE conditioning: FIF splits the signal into intrinsic mode components
    whose instantaneous frequency is well behaved, so the component carrying
    the oscillation of interest is mono-component by construction rather than
    merely band-limited. Returns None if FIF is unavailable or fails so the
    caller can fall back to the bandpassed signal.
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

    JADE's zero-crossing detector is a bare sign-change test with no noise
    rejection of its own; feeding it a wideband signal lets sample-to-sample
    noise (comparable in size to the true per-sample step near a zero-crossing
    when fs >> f0) inject spurious crossings and corrupt the DTW cycle fits. So
    JADE is always run on a conditioned, mono-component signal:

      * if `use_fif` and `raw` is given, `raw` is decomposed with Fast
        Iterative Filtering and the IMC nearest the target frequency (`f0`, or
        the mean of `true_inst_freq`) is used;
      * otherwise `filtered` is used, which must already be bandpass-filtered
        (e.g. compute_hilbert_reference's output).
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

    # JADE's phase is zero-crossing-referenced (x ~ sign*amp*sin(2*pi*phase));
    # true_phase is peak-referenced (x ~ amp*cos(phase)). Convert and resolve
    # the sign from JADE's own reconstruction, same as in PHASE_estimation/phase_test.py.
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
        label = r"$\hat \theta - \theta$" if true_phase is not None else r"$\hat \theta - \theta_{\mathrm{HT}}$"
        err = np.angle(np.exp(1j * (phi - ref)), deg=True)
        errors.append({"label": label, "time_s": t, "values": err,
                        "unit": "degrees", "linestyle": "-"})

        if true_phase is not None:
            h_err = np.angle(np.exp(1j * (hilbert_phase[:n] - true_phase[:n])), deg=True)
            errors.append({"label": r"$\theta_{\mathrm{HT}} - \theta$", "time_s": t, "values": h_err,
                           "unit": "degrees", "linestyle": "-"})

            # Online estimate vs the offline Hilbert estimate, as if Hilbert
            # were ground truth. When there's no true_phase this is identical
            # to "Phase error", so it's only added here to avoid a duplicate.
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

        # Ideal stimulus train -- kept only for visualisation / EDF export;
        # the edge errors below no longer match against it.
        stim_ref, _, _ = compute_reference_stimulus(ref_phase, f0_y, fs, stim_cfg)
        n_ideal  = len(get_edges(stim_ref, "rising"))
        n_actual = len(get_edges(trigger_binary, "rising"))
        if n_actual != n_ideal:
            print(f"Stimulus: {n_actual} onset(s) delivered vs {n_ideal} ideal.")

        # Phase error at each trigger edge vs the phase the controller targets,
        # scored against the offline reference phase (true_phase if available,
        # else the offline Hilbert phase).
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

        # With stim_dur_unit == "ms" there's no target phase for the falling
        # edge (the burst is timed, not phase-targeted), so offset_err above
        # is None; score the actual-vs-target burst length in ms instead.
        if stim_cfg.stim_dur_unit != "deg":
            dur_err = compute_stimulus_duration_error(trigger_binary, time_us, stim_cfg, start_ts)
            if dur_err is not None:
                print(f"Stim duration error: {np.nanmean(dur_err['values']):.2f} ms "
                      f"+- {np.nanstd(dur_err['values']):.2f} ms")
                errors.append({"label": "Stim duration error",
                               "time_s": dur_err["time_s"], "values": dur_err["values"],
                               "unit": "ms"})

        # Same edges again, but scored against the online phase estimate
        # (`PhaseEstimation_phase`, produced live by the real-time ECHT-based
        # PhaseEstimation processor) instead of the offline/ground-truth
        # reference above -- i.e. what the running system actually saw,
        # rather than an offline reconstruction.
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
