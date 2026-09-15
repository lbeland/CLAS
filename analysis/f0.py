"""
Fundamental frequency (f0) estimation.

"""

import numpy as np
from scipy.signal import welch, savgol_filter, medfilt
from scipy.stats import linregress
from tqdm import tqdm

from .modal import modal
from .plot import _edge_trim_window


def gaussian_peak(freqs, amp, center, width):
    return amp * np.exp(-0.5 * ((freqs - center) / width) ** 2)


def bic_peak_test_simple(freqs, residual, gaussian, fmin, fmax, floor_value):
    fit_mask = (freqs >= fmin) & (freqs <= fmax)
    fit_resid = residual[fit_mask]
    n = len(fit_resid)

    if n < 4:
        return False, 0.0

    ss_h0 = np.sum((fit_resid - floor_value) ** 2)
    ss_h1 = np.sum((fit_resid - gaussian[fit_mask]) ** 2)
    bic_h0 = n * np.log(max(ss_h0, np.nextafter(0, 1)) / n)
    bic_h1 = n * np.log(max(ss_h1, np.nextafter(0, 1)) / n) + 3 * np.log(n)  # 3 params for Gaussian (amp, center, width)
    return (bic_h0 - bic_h1) > 0, bic_h0 - bic_h1


def approve_peak(freqs, psd_safe, psd_flat, psd_smooth, popt, gaussian, aperiodic_simple, alpha_band, floor_value):
    fmin, fmax = freqs[0], freqs[-1]
    peak_sig, _delta_bic = bic_peak_test_simple(freqs, psd_flat, gaussian, fmin, fmax, floor_value)
    return bool(peak_sig)


def alpha_fast(psd, freq_bins, config):
    """Whitens the spectrum against a robust aperiodic fit, smooths it, and fits
    a Gaussian to the resulting peak to estimate the fundamental frequency. The
    whitened spectrum, its smoothed version and the Gaussian all stay in
    log10-ratio space (0.0 = exactly on the aperiodic fit), which keeps
    amplitude / gaussian / floor_value on a consistent scale for the BIC test in
    ``approve_peak``.

    Returns ``(est_pf, [slope, intercept], snr)`` -- the peak frequency (Hz),
    the refined aperiodic-fit parameters, and the linear-scale signal-to-noise
    ratio of the approved alpha peak. When no peak is found or it fails the BIC
    test, ``est_pf`` and ``snr`` are ``np.nan`` but ``[slope, intercept]`` still
    holds the refined aperiodic fit (it doesn't depend on a peak existing), so
    callers that only want the 1/f fit -- e.g. spectral whitening -- can still
    use it. ``[slope, intercept]`` is ``[np.nan, np.nan]`` only if the linear
    fit itself degenerates (too few sub-fit points).

    ``snr`` is computed exactly as in the online ``FrequencyEstimation``
    processor (``extensions/processors/FrequencyEstimation/FrequencyEstimation.cpp``):
    over the bins within +/-2 sigma of the peak, the aperiodic fit is taken
    back to linear power (``10 ** aperiodic``), the smoothed Gaussian bump is
    taken to a linear ratio above that fit (``10 ** gaussian - 1``), and
    ``snr = sum(aperiodic_lin * gauss_ratio) / sum(aperiodic_lin)``.
    """
    band = (freq_bins >= config["freq_range"][0]) & (freq_bins <= config["freq_range"][1])
    psd_band = psd[band]
    freqs = freq_bins[band]
    log_freqs = np.log10(freqs)

    resolution = freqs[1] - freqs[0]
    sav_gol_window_length = int(2.5 / resolution)
    if sav_gol_window_length % 2 == 0:
        sav_gol_window_length += 1

    sav_gol_polyorder = 3
    if sav_gol_polyorder >= sav_gol_window_length:
        sav_gol_window_length = sav_gol_polyorder + 2

    eps = np.nextafter(0, 1)  # Smallest positive float
    log_psd = np.log10(np.maximum(psd_band, eps))
    slope, intercept, _, _, _ = linregress(log_freqs, log_psd)
    aperiodic_initial = log_freqs * slope + intercept  # log10-scale, first (unrefined) fit
    psd_flat = log_psd - aperiodic_initial  # log10 ratio: 0.0 = on fit

    # Select only samples that are exactly on (0.0) or below the perfect fit
    ratio_threshold = 0.0
    mask = psd_flat <= ratio_threshold
    freqs_refit = log_freqs[mask]
    psd_refit = log_psd[mask]
    # Re-fit using only the selected samples to ignore peak oscillations
    slope, intercept, _, _, _ = linregress(freqs_refit, psd_refit)
    aperiodic_simple = log_freqs * slope + intercept  # log10-scale, refined fit
    psd_flat = log_psd - aperiodic_simple  # whitened spectrum: log10 ratio, 0.0 = on fit

    psd_smooth = savgol_filter(psd_flat, window_length=sav_gol_window_length, polyorder=sav_gol_polyorder)

    fmin, fmax = config["alpha_band"][0], config["alpha_band"][1]

    alpha_band = (freqs >= fmin) & (freqs <= fmax)

    psd_alpha_band = psd_smooth[alpha_band]

    max_bin = np.argmax(psd_alpha_band)  # index relative to the alpha_band subset
    est_pf = freqs[alpha_band][max_bin]

    if 0 < max_bin < (psd_alpha_band.size - 1):
        y1, y2, y3 = psd_alpha_band[max_bin - 1], psd_alpha_band[max_bin], psd_alpha_band[max_bin + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            est_pf += delta * resolution

    floor_value = 0.0  # log10-ratio baseline: 0.0 = exactly on the aperiodic fit
    global_max_bin = max_bin + np.where(alpha_band)[0][0]  # index into the full freqs/psd_smooth arrays

    amplitude = psd_smooth[global_max_bin]  # log10-ratio value of the smoothed spectrum at the peak bin

    # Reject outright if the smoothed peak doesn't clear the aperiodic fit
    # (log-ratio <= 0) -- there's no bump to fit a Gaussian to.
    if amplitude <= 0.0:
        return np.nan, [slope, intercept], np.nan

    # FWHM: walk outward from the peak bin, in each direction, until the
    # smoothed spectrum drops below half-max (over the full spectrum, not
    # just the alpha band).
    half_max = amplitude / 2.0
    left_bin = global_max_bin
    while left_bin > 0 and psd_smooth[left_bin] > half_max:
        left_bin -= 1
    right_bin = global_max_bin
    while right_bin < psd_smooth.size - 1 and psd_smooth[right_bin] > half_max:
        right_bin += 1
    fwhm = max((right_bin - left_bin) * resolution, resolution)
    std_gauss = fwhm / (2 * np.sqrt(2 * np.log(2)))

    if std_gauss > 2:
        return np.nan, [slope, intercept], np.nan

    popt = [amplitude, est_pf, std_gauss]

    gaussian = gaussian_peak(freqs, *popt) + floor_value

    peak_approved = approve_peak(freqs, log_psd, psd_flat, psd_smooth, popt, gaussian,
                                 aperiodic_simple, config["alpha_band"], floor_value=floor_value)

    if not peak_approved:
        return np.nan, [slope, intercept], np.nan

    # SNR of the approved peak, matching FrequencyEstimation.cpp: over the bins
    # within +/-2 sigma of the peak, weight the linear-scale aperiodic power by
    # the linear-scale Gaussian ratio above the fit (10**gaussian - 1), and
    # normalise by the summed aperiodic power.
    snr_band = np.abs(freqs - est_pf) <= 2 * std_gauss
    aperiodic_lin = 10.0 ** aperiodic_simple[snr_band]
    gauss_ratio = 10.0 ** gaussian[snr_band] - 1.0
    signal_power = np.sum(aperiodic_lin * gauss_ratio)
    noise_power = np.sum(aperiodic_lin)
    snr = signal_power / max(noise_power, eps)

    return est_pf, [slope, intercept], snr


def estimate_f0(raw: np.ndarray, fs: float) -> list:
    """Estimate f0 from the raw signal. Returns a list of per-window estimates."""
    config = {"alpha_band": (5, 18), "freq_range": (0.01, 30.0)}

    if len(raw) / fs > 20:
        window_samples = int(20 * fs)
        starts = list(range(0, len(raw) - window_samples, window_samples))
        last_start = len(raw) - window_samples
        if not starts or starts[-1] != last_start:
            starts.append(last_start)  # cover the trailing remainder instead of dropping it
        f0_windowed = []
        aperiodic_params_windowed = []
        snr_windowed = []
        for start in tqdm(starts):
            # freqs, psd = welch(raw[start : start + window_samples], fs=fs, nperseg=int(fs * 10))
            freqs = np.fft.rfftfreq(len(raw[start : start + window_samples]), d=1/fs)
            X = np.fft.rfft(raw[start : start + window_samples])
            psd = (np.abs(X) ** 2) / (fs * len(raw[start : start + window_samples]))
            psd[1:-1] *= 2
            f0, aperiodic_params, snr = alpha_fast(psd, freqs, config)
            f0_windowed.append(f0)
            aperiodic_params_windowed.append(aperiodic_params)
            snr_windowed.append(snr)
        return f0_windowed, np.nanmean(np.array(aperiodic_params_windowed), axis=0), np.nanmean(np.array(snr_windowed))  # mean aperiodic fit across windows (alpha_fast returns one even with no approved peak; nanmean still guards degenerate fits)

    else:
        # freqs, psd = welch(raw, fs=fs, nperseg=int(fs * 30))
        freqs = np.fft.rfftfreq(len(raw), d=1/fs)
        X = np.fft.rfft(raw)
        psd = (np.abs(X) ** 2) / (fs * len(raw))
        psd[1:-1] *= 2
        f0, aperiodic_params, snr = alpha_fast(psd, freqs, config)
    return [f0], np.array(aperiodic_params), snr


def estimate_f0_with_phase(hilbert_phase: np.ndarray, fs: float, f0: float) -> np.ndarray:
    """
    Estimate f0 from the instantaneous phase of the Hilbert transform, using the
    multi-order median filtering ("frequency sliding") technique of Cohen: the
    noisy instantaneous frequency is median filtered at 10 window sizes spanning
    10-400 ms, and the median across those filtered estimates is taken as the
    final f0 time series.

    Since the estimate is derived from the Hilbert phase, it is subject to the
    same edge transients as other phase-estimate metrics, so the leading and
    trailing edges are trimmed to NaN with the same window used for those
    (see _edge_trim_window()).
    """
    inst_freq = np.diff(np.unwrap(hilbert_phase), prepend=f0) * fs / (2 * np.pi)

    n_orders = 10
    half_widths = np.round(np.linspace(10, 400, n_orders) / 2 / 1000 * fs).astype(int)

    filtered = np.empty((n_orders, len(inst_freq)))
    for i, half_width in enumerate(half_widths):
        filtered[i] = medfilt(inst_freq, kernel_size=2 * half_width + 1)

    result = np.median(filtered, axis=0)

    time_s = np.arange(len(result)) / fs
    t_lo, t_hi = _edge_trim_window([{"time_s": time_s}])
    result[(time_s < t_lo) | (time_s > t_hi)] = np.nan

    return result


def estimate_f0_modal(raw: np.ndarray, fs: float, alpha_band: tuple = (5, 18),
                      freq_range: tuple = (1.0, 30.0)) -> np.ndarray:
    """Estimate a continuous f0 time series with MODAL (Watrous 2017): bands
    are adaptively identified from where the signal's power exceeds a robust
    1/f background fit, then instantaneous frequency ("frequency sliding",
    Cohen 2014) is tracked within each band. Of the detected bands, the one
    closest to the centre of alpha_band is returned, for comparison against
    estimate_f0_with_phase()'s Hilbert-phase estimate.

    Returns a NaN-filled array the length of raw if MODAL finds no band
    overlapping alpha_band.
    """
    params = {
        "srate": fs,
        "wavefreqs": np.arange(freq_range[0], freq_range[1], 0.5),
        "local_winsize_sec": [],
        "wavecycles": 6,
        "crop_fs": True,
    }
    frequency_sliding = modal(raw, params)

    if frequency_sliding.ndim != 2:
        return np.full(len(raw), np.nan, dtype=np.float32)

    band_means = np.nanmean(frequency_sliding, axis=1)
    in_alpha = np.where((band_means >= alpha_band[0]) & (band_means <= alpha_band[1]))[0]
    if len(in_alpha) == 0:
        return np.full(len(raw), np.nan, dtype=np.float32)

    center = (alpha_band[0] + alpha_band[1]) / 2
    best = in_alpha[np.argmin(np.abs(band_means[in_alpha] - center))]
    return frequency_sliding[best]
