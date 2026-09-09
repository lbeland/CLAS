"""
Fundamental frequency (f0) estimation.

"""

import numpy as np
from scipy.signal import welch, savgol_filter, medfilt
from scipy.stats import linregress
from tqdm import tqdm


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

    Returns ``(est_pf, [slope, intercept])`` -- the peak frequency (Hz) and the
    refined aperiodic-fit parameters. When no peak is found or it fails the BIC
    test, ``est_pf`` is ``np.nan`` but ``[slope, intercept]`` still holds the
    refined aperiodic fit (it doesn't depend on a peak existing), so callers
    that only want the 1/f fit -- e.g. spectral whitening -- can still use it.
    ``[slope, intercept]`` is ``[np.nan, np.nan]`` only if the linear fit itself
    degenerates (too few sub-fit points).
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
        return np.nan, [slope, intercept]

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
        return np.nan, [slope, intercept]

    popt = [amplitude, est_pf, std_gauss]

    gaussian = gaussian_peak(freqs, *popt) + floor_value

    peak_approved = approve_peak(freqs, log_psd, psd_flat, psd_smooth, popt, gaussian,
                                 aperiodic_simple, config["alpha_band"], floor_value=floor_value)

    if not peak_approved:
        return np.nan, [slope, intercept]

    return est_pf, [slope, intercept]


def estimate_f0(raw: np.ndarray, fs: float) -> list:
    """Estimate f0 from the raw signal. Returns a list of per-window estimates."""
    config = {"alpha_band": (5, 18), "freq_range": (0.01, 30.0)}

    if len(raw) / fs > 30:
        window_samples = int(30 * fs)
        starts = list(range(0, len(raw) - window_samples, window_samples))
        last_start = len(raw) - window_samples
        if not starts or starts[-1] != last_start:
            starts.append(last_start)  # cover the trailing remainder instead of dropping it
        f0_windowed = []
        aperiodic_params_windowed = []
        for start in tqdm(starts):
            # freqs, psd = welch(raw[start : start + window_samples], fs=fs, nperseg=int(fs * 10))
            freqs = np.fft.rfftfreq(len(raw[start : start + window_samples]), d=1/fs)
            X = np.fft.rfft(raw[start : start + window_samples])
            psd = (np.abs(X) ** 2) / (fs * len(raw[start : start + window_samples]))
            psd[1:-1] *= 2
            f0, aperiodic_params = alpha_fast(psd, freqs, config)
            f0_windowed.append(f0)
            aperiodic_params_windowed.append(aperiodic_params)
        return f0_windowed, np.nanmean(np.array(aperiodic_params_windowed), axis=0)  # mean aperiodic fit across windows (alpha_fast returns one even with no approved peak; nanmean still guards degenerate fits)

    else:
        # freqs, psd = welch(raw, fs=fs, nperseg=int(fs * 30))
        freqs = np.fft.rfftfreq(len(raw), d=1/fs)
        X = np.fft.rfft(raw)
        psd = (np.abs(X) ** 2) / (fs * len(raw))
        psd[1:-1] *= 2
        f0, aperiodic_params = alpha_fast(psd, freqs, config)
    return [f0], np.array(aperiodic_params)


def estimate_f0_with_phase(hilbert_phase: np.ndarray, fs: float, f0: float) -> np.ndarray:
    """
    Estimate f0 from the instantaneous phase of the Hilbert transform, using the
    multi-order median filtering ("frequency sliding") technique of Cohen: the
    noisy instantaneous frequency is median filtered at 10 window sizes spanning
    10-400 ms, and the median across those filtered estimates is taken as the
    final f0 time series.
    """
    inst_freq = np.diff(np.unwrap(hilbert_phase), prepend=f0) * fs / (2 * np.pi)

    n_orders = 10
    half_widths = np.round(np.linspace(10, 400, n_orders) / 2 / 1000 * fs).astype(int)

    filtered = np.empty((n_orders, len(inst_freq)))
    for i, half_width in enumerate(half_widths):
        filtered[i] = medfilt(inst_freq, kernel_size=2 * half_width + 1)

    return np.median(filtered, axis=0)
