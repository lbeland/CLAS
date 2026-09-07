"""
Signal analysis: Hilbert reference and error computation.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import ellip, butter, firwin, filtfilt, sosfiltfilt, hilbert, windows
from tqdm import tqdm

from .stimulus import StimulusConfig, compute_reference_stimulus, compute_stimulus_edge_errors, get_edges
from .f0 import combine_simple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PHASE_estimation.jade import jade_v3
from IAF_Estimation.cecHT.phase import ECHT


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


_F0_CONFIG = {"alpha_band": (5, 18), "freq_range": (0.01, 30.0)}


def compute_echt_reference(
    filtered: np.ndarray,
    f0_series: np.ndarray | None,
    fs: float,
    eval_idx: np.ndarray,
    freq_window_s: float = 10.0,
    n_cycles: float = 2.0,
    bw_factor: float = 0.9,
) -> np.ndarray:
    """Sparse offline phase reference: a calibrated, endpoint-corrected Hilbert
    transform (ECHT) whose centre frequency is re-estimated locally, evaluated
    only at the caller-chosen samples ``eval_idx`` (here: every stimulus onset).

    For every sample ``x`` in ``eval_idx``:

      1. estimate the dominant alpha frequency ``f0(x)`` with
         :func:`analysis.f0_series.combine_simple` on the periodogram of the
         ``freq_window_s``-second slice of ``filtered`` *centred* on ``x``;
      2. take the calibrated ECHT (see ``IAF_Estimation/cecHT/phase.py``) of
         the ``n_cycles``-cycle slice of ``filtered`` *ending at* ``x``, with
         ``f0(x)`` as the calibration centre frequency and an order-1
         Butterworth pass-band ``f0(x) * (1 ± bw_factor / 2)``;
      3. the ECHT phase at that window's endpoint is the reference phase at ``x``.

    Window length (``n_cycles / f0`` s), pass-band width (``0.9 * f0``), filter
    order and MSE calibration all match the online ``PhaseEstimation`` processor.
    ``f0(x)`` is snapped to a 0.1 Hz grid, as the online processor also does.

    Returns
    -------
    echt_phase : np.ndarray, shape ``(len(filtered),)``
        Peak-referenced, wrapped reference phase (rad) at each evaluated
        sample; ``np.nan`` everywhere else (including any ``eval_idx`` sample
        where ``combine_simple`` approved no alpha peak or a window ran off a
        recording edge).
    """
    filtered = np.asarray(filtered, dtype=float)
    n  = len(filtered)
    fs = float(fs)
    eval_idx = np.unique(np.asarray(eval_idx, dtype=int))
    eval_idx = eval_idx[(eval_idx >= 0) & (eval_idx < n)]

    echt_phase = np.full(n, np.nan)
    if eval_idx.size == 0:
        return echt_phase

    half_freq = int(round(freq_window_s * fs / 2.0))
    n_nopeak = n_edge = 0
    for x in tqdm(eval_idx, desc="ECHT reference"):
        if x - half_freq < 0 or x + half_freq > n:
            n_edge += 1
            continue

        if f0_series is not None:
            # Use the local f0 estimate if available, rather than a global
            # combine_simple() call, to avoid spurious peaks in the periodogram
            # from corrupting the ECHT reference.
            f0 = f0_series[x]
            if not np.isfinite(f0):
                n_nopeak += 1
                continue
        else:
            seg   = filtered[x - half_freq : x + half_freq]
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
            X     = np.fft.rfft(seg)
            psd   = (np.abs(X) ** 2) / (fs * len(seg))
            psd[1:-1] *= 2
            try:
                f0, _ = combine_simple(psd, freqs, _F0_CONFIG)
            except Exception:
                f0 = np.nan
            if not np.isfinite(f0):
                n_nopeak += 1
                continue                  # 0.1 Hz grid, like the online processor

        bw     = bw_factor * f0
        l_freq = f0 - bw / 2.0
        h_freq = f0 + bw / 2.0
        w      = int(round(n_cycles * fs / f0))
        if l_freq <= 0.0 or h_freq >= fs / 2.0 or w < 3 or x - w + 1 < 0:
            n_edge += 1
            continue

        echt = ECHT(l_freq=l_freq, h_freq=h_freq, sfreq=fs, filt_order=1,
                    filter_type="butter", calibrate=True, f0=f0, fft_mode="fast")
        z = np.squeeze(echt.fit_transform(filtered[x - w + 1 : x + 1]))
        echt_phase[x] = np.angle(z[-1])

    n_ok = int(np.isfinite(echt_phase[eval_idx]).sum())
    print(f"compute_echt_reference: {n_ok}/{eval_idx.size} onset(s) evaluated "
          f"({n_nopeak} without an approved peak, {n_edge} skipped at edges)")
    return echt_phase


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
    filtered: "np.ndarray | None" = None,
) -> tuple[list[dict], np.ndarray | None]:
    """
    Compute phase, f0, and stimulus edge errors.
    Returns (list of error dicts, stim_ref array or None).
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
        # scored against the offline Hilbert/true reference and -- for onsets
        # only -- also against the per-onset ECHT reference (compute_echt_reference).
        onset_err, offset_err = compute_stimulus_edge_errors(
            trigger_binary, time_us, ref_phase, f0_y, stim_cfg, start_ts)
        if onset_err is not None:
            errors.append({"label": "Stim onset error",
                           "time_s": onset_err["time_s"], "values": onset_err["values"],
                           "unit": "degrees"})
        if offset_err is not None:
            errors.append({"label": "Stim offset error",
                           "time_s": offset_err["time_s"], "values": offset_err["values"],
                           "unit": "degrees", "linestyle": "--"})

        if filtered is not None:
            filt_signal = filtered
            f0 = f0_y if samples.get("FrequencyEstimation") is not None else None
            echt_phase = compute_echt_reference(filt_signal, f0, fs, get_edges(trigger_binary, "rising"))
            echt_onset_err, _ = compute_stimulus_edge_errors(
                trigger_binary, time_us, echt_phase, f0_y, stim_cfg, start_ts)
            if echt_onset_err is not None:
                errors.append({"label": "Stim onset error (ECHT)",
                               "time_s": echt_onset_err["time_s"], "values": echt_onset_err["values"],
                               "unit": "degrees", "linestyle": ":"})

    return errors, stim_ref
