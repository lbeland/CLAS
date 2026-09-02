"""Simple (aperiodic-subtracted, Savgol-smoothed) IAF estimator.

`combine_simple` and its helpers (`gaussian_peak`, `bic_peak_test_simple`,
`approve_peak`) are copied **verbatim** from
``IAF_Estimation/IAF_tests.py`` (the ``combine_simple`` branch of
``run_algorithms``) so the pipeline's ``--iaf-method simple`` uses exactly that
algorithm. Keep them byte-for-byte in sync with IAF_tests.py.

`simple_paf` is the thin adapter the EEG pipeline calls: it builds the same
one-sided periodogram PSD that ``IAF_tests.run_window_analysis`` feeds to
``combine_simple`` and returns the peak alpha frequency (or ``np.nan``).
"""

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter


# --- verbatim from IAF_tests.py -------------------------------------------------
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
    peak_sig, delta_bic = bic_peak_test_simple(freqs, psd_flat, gaussian, fmin, fmax, floor_value)

    if not peak_sig:
        return False
    else:
        return True


def combine_simple(psd, freq_bins, config):
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
    slope, intercept, _, _, _ = stats.linregress(log_freqs, log_psd)
    aperiodic_simple = log_freqs * slope + intercept  # log10-scale
    # psd_flat = psd_safe / np.power(10, aperiodic_simple)  # ratio: 1.0 = on fit
    psd_flat = log_psd - aperiodic_simple  # log10 ratio: 0.0 = on fit

    # Select only samples that are exactly on (1) or below the perfect fit
    ratio_threshold = 0.0  # 1.0
    mask = psd_flat <= ratio_threshold
    freqs_refit = log_freqs[mask]
    psd_refit = log_psd[mask]
    # Re-fit using only the selected samples to ignore peak oscillations
    slope, intercept, _, _, _ = stats.linregress(freqs_refit, psd_refit)
    aperiodic_simple = log_freqs * slope + intercept  # log10-scale
    # psd_flat = psd_safe / np.power(10, aperiodic_simple)
    psd_flat = np.power(10, np.maximum(log_psd - aperiodic_simple, eps))  # log10 ratio: 0.0 = on fit

    psd_smooth = savgol_filter(psd_flat, window_length=sav_gol_window_length, polyorder=sav_gol_polyorder)

    fmin, fmax = config["alpha_band"][0], config["alpha_band"][1]

    alpha_band = (freqs >= fmin) & (freqs <= fmax)

    psd_alpha_band = psd_smooth[alpha_band]
    fit_freqs = freqs[alpha_band]

    max_bin = np.argmax(psd_alpha_band)
    est_pf = freqs[alpha_band][max_bin]

    if 0 < max_bin < (psd_alpha_band.size - 1):
        y1, y2, y3 = psd_alpha_band[max_bin - 1], psd_alpha_band[max_bin], psd_alpha_band[max_bin + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            est_pf += delta * resolution

    floor_value = 0  # 1

    amp_guess = psd_smooth[alpha_band][max_bin] - floor_value
    # center_guess = fit_freqs[max_bin]

    half_max = amp_guess / 2 + floor_value
    global_max_bin = max_bin + np.where(alpha_band)[0][0]
    left_idx = np.where(psd_smooth[:global_max_bin] < half_max)[0]
    if len(left_idx) > 0:
        left_idx = left_idx[-1]
    else:
        left_idx = global_max_bin
    right_idx = np.where(psd_smooth[alpha_band][global_max_bin:] < half_max)[0]
    if len(right_idx) > 0:
        right_idx = right_idx[0] + global_max_bin
    else:
        right_idx = global_max_bin
    fwhm = max((right_idx - left_idx) * resolution, resolution)
    std_gauss = fwhm / (2 * np.sqrt(2 * np.log(2)))
    if std_gauss > 2:
        # print(std_gauss)
        return np.nan
    popt = [amp_guess, est_pf, std_gauss]

    gaussian = gaussian_peak(freqs, *popt) + floor_value

    peak_approved = approve_peak(freqs, log_psd, psd_flat, psd_smooth, popt, gaussian, aperiodic_simple, config["alpha_band"], floor_value=floor_value)

    if not peak_approved:
        return np.nan

    return est_pf
# --- end verbatim -------------------------------------------------------------


def _periodogram(window, fs):
    """One-sided periodogram PSD, identical to IAF_tests.run_window_analysis."""
    window = np.asarray(window, dtype=float)
    n = window.size
    freq_bins = np.fft.rfftfreq(n, 1 / fs)
    X = np.fft.rfft(window, n=n)
    psd = (np.abs(X) ** 2) / (fs * n)
    psd[1:-1] *= 2  # one-sided correction (except DC and Nyquist)
    return freq_bins, psd


def simple_paf(window, fs, alpha_band=(7.5, 14.0), freq_range=(0.1, 30.0)):
    """Peak alpha frequency of ``window`` via :func:`combine_simple`.

    Parameters
    ----------
    window : 1-D array   raw signal samples for one IAF window
    fs : float           sampling rate (Hz)
    alpha_band : (float, float)   alpha search edges (Hz)
    freq_range : (float, float)   broadband range for the 1/f fit (Hz)

    Returns
    -------
    float   estimated PAF in Hz, or ``np.nan`` if no peak is approved.
    """
    freq_bins, psd = _periodogram(window, fs)
    config = {"freq_range": tuple(freq_range), "alpha_band": tuple(alpha_band)}
    try:
        return float(combine_simple(psd, freq_bins, config))
    except Exception:  # noqa: BLE001 - match the pipeline's per-window robustness
        return np.nan
