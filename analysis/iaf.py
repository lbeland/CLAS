"""
Individual alpha frequency (IAF) estimation.
"""

import numpy as np
from scipy.signal import welch, savgol_filter, medfilt
from scipy.stats import linregress
from tqdm import tqdm
# from fooof import FOOOF


def gaussian_peak(freqs, amp, center, width):
    return amp * np.exp(-0.5 * ((freqs - center) / width) ** 2)


def bic_peak_test_simple(
    freqs: np.ndarray,
    residual: np.ndarray,
    gaussian: np.ndarray,
    fmin: float,
    fmax: float,
    floor_value: float,
) -> tuple[bool, float]:
    fit_mask  = (freqs >= fmin) & (freqs <= fmax)
    fit_resid = residual[fit_mask]
    n = len(fit_resid)
    if n < 4:
        return False, 0.0

    eps   = np.nextafter(0, 1)
    ss_h0 = np.sum((fit_resid - floor_value) ** 2)
    ss_h1 = np.sum((fit_resid - gaussian[fit_mask]) ** 2)
    bic_h0 = n * np.log(max(ss_h0, eps) / n)
    bic_h1 = n * np.log(max(ss_h1, eps) / n) + 3 * np.log(n)
    return (bic_h0 - bic_h1) > 0, bic_h0 - bic_h1


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
    slope, intercept, _, _, _ = linregress(log_freqs, log_psd)
    aperiodic_simple = log_freqs * slope + intercept  # log10-scale
    # psd_flat = psd_safe / np.power(10, aperiodic_simple)  # ratio: 1.0 = on fit
    psd_flat = log_psd - aperiodic_simple  # log10 ratio: 0.0 = on fit

    # Select only samples that are exactly on (1) or below the perfect fit
    ratio_threshold = 0.0 # 1.0
    mask = psd_flat <= ratio_threshold
    freqs_refit = log_freqs[mask]
    psd_refit = log_psd[mask]
    # Re-fit using only the selected samples to ignore peak oscillations
    slope, intercept, _, _, _ = linregress(freqs_refit, psd_refit)
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


    floor_value = 0 #1

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
    fwhm = max((right_idx-left_idx) * resolution, resolution)
    std_gauss = fwhm / (2 * np.sqrt(2 * np.log(2)))
    if std_gauss > 2:
        # print(std_gauss)
        return np.nan
    popt = [amp_guess, est_pf, std_gauss]

    gaussian = gaussian_peak(freqs, *popt) + floor_value
    
    peak_approved = bic_peak_test_simple(freqs, psd_flat, gaussian, fmin, fmax, floor_value)

    if not peak_approved:
        return np.nan, [np.nan, np.nan]

    return est_pf, [slope, intercept]


def estimate_iaf(raw: np.ndarray, fs: float) -> list:
    """Estimate IAF from the raw signal. Returns a list of per-window estimates."""
    config = {"alpha_band": (5, 18), "freq_range": (0.01, 30.0)}

    if len(raw) / fs > 20:
        window_samples = int(20 * fs)
        starts = list(range(0, len(raw) - window_samples, window_samples))
        last_start = len(raw) - window_samples
        if not starts or starts[-1] != last_start:
            starts.append(last_start)  # cover the trailing remainder instead of dropping it
        iaf_windowed = []
        aperiodic_params_windowed = []
        for start in tqdm(starts):
            freqs, psd = welch(raw[start : start + window_samples], fs=fs, nperseg=int(fs * 10))
            iaf, aperiodic_params = combine_simple(psd, freqs, config)
            iaf_windowed.append(iaf)
            aperiodic_params_windowed.append(aperiodic_params)
        return iaf_windowed, np.mean(np.array(aperiodic_params_windowed), axis=0)  # Return mean aperiodic params across windows
    
    else:
        freqs, psd = welch(raw, fs=fs, nperseg=int(fs * 10))
        iaf, aperiodic_params = combine_simple(psd, freqs, config)
        return [iaf], np.array(aperiodic_params)
        

def estimate_iaf_with_phase(hilbert_phase: np.ndarray, fs: float, f0: float) -> np.ndarray:
    """
    Estimate IAF from the instantaneous phase of the Hilbert transform, using the
    multi-order median filtering ("frequency sliding") technique of Cohen: the
    noisy instantaneous frequency is median filtered at 10 window sizes spanning
    10-400 ms, and the median across those filtered estimates is taken as the
    final IAF time series.
    """
    inst_freq = np.diff(np.unwrap(hilbert_phase), prepend=f0) * fs / (2 * np.pi)

    n_orders = 10
    half_widths = np.round(np.linspace(10, 400, n_orders) / 2 / 1000 * fs).astype(int)

    filtered = np.empty((n_orders, len(inst_freq)))
    for i, half_width in enumerate(half_widths):
        filtered[i] = medfilt(inst_freq, kernel_size=2 * half_width + 1)

    return np.median(filtered, axis=0)