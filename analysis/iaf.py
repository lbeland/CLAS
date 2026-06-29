"""
Individual alpha frequency (IAF) estimation.
"""

import numpy as np
from scipy.signal import welch, savgol_filter
from scipy.stats import linregress
from tqdm import tqdm
from fooof import FOOOF


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


def combine_simple(psd: np.ndarray, freq_bins: np.ndarray, config: dict) -> float:
    """Fit aperiodic component and return PAF; returns np.nan if no clear peak."""
    band = (freq_bins >= config["freq_range"][0]) & (freq_bins <= config["freq_range"][1])
    psd_band = psd[band]
    freqs    = freq_bins[band]

    resolution = freqs[1] - freqs[0]
    win = max(int(2.5 / resolution) | 1, 5)   # odd, at least 5
    polyorder = 3
    if polyorder >= win:
        win = polyorder + 2

    eps      = np.nextafter(0, 1)
    psd_safe = np.maximum(psd_band, eps)

    slope, intercept, _, _, _ = linregress(np.log10(freqs), np.log10(psd_safe))
    aperiodic = np.log10(freqs) * slope + intercept
    psd_flat  = psd_safe / np.power(10, aperiodic)

    # Re-fit using only sub-peak samples to reduce peak influence
    mask = psd_flat <= 1.0
    if mask.sum() >= 2:
        slope, intercept, _, _, _ = linregress(np.log10(freqs[mask]), np.log10(psd_safe[mask]))
        aperiodic = np.log10(freqs) * slope + intercept
        psd_flat  = psd_safe / np.power(10, aperiodic)

    psd_smooth = savgol_filter(psd_flat, window_length=win, polyorder=polyorder)

    fmin, fmax = config["alpha_band"]
    alpha_band     = (freqs >= fmin) & (freqs <= fmax)
    psd_alpha      = psd_smooth[alpha_band]
    max_bin        = np.argmax(psd_alpha)
    paf            = freqs[alpha_band][max_bin]

    # Parabolic interpolation for sub-bin accuracy
    if 0 < max_bin < psd_alpha.size - 1:
        y1, y2, y3 = psd_alpha[max_bin - 1], psd_alpha[max_bin], psd_alpha[max_bin + 1]
        denom = y1 - 2 * y2 + y3
        if denom != 0:
            paf += 0.5 * (y1 - y3) / denom * resolution

    floor_value = 1.0
    amp_guess   = psd_alpha[max_bin] - floor_value
    center_guess = freqs[alpha_band][max_bin]
    half_max     = amp_guess / 2 + floor_value
    g_max_bin    = max_bin + np.where(alpha_band)[0][0]

    left_idx  = np.where(psd_smooth[:g_max_bin] < half_max)[0]
    left_idx  = left_idx[-1]  if len(left_idx)  > 0 else g_max_bin
    right_idx = np.where(psd_smooth[alpha_band][g_max_bin:] < half_max)[0]
    right_idx = right_idx[0] + g_max_bin if len(right_idx) > 0 else g_max_bin

    fwhm      = max((right_idx - left_idx) * resolution, resolution)
    std_gauss = fwhm / (2 * np.sqrt(2 * np.log(2)))
    if std_gauss > 2:
        return np.nan

    gauss  = gaussian_peak(freqs, amp_guess, center_guess, std_gauss) + floor_value
    approved, _ = bic_peak_test_simple(freqs, psd_flat, gauss, freqs[0], freqs[-1], floor_value)
    return paf if approved else np.nan


def estimate_iaf(raw: np.ndarray, fs: float) -> list:
    """Estimate IAF from the raw signal. Returns a list of per-window estimates."""
    config = {"alpha_band": (5, 18), "freq_range": (0.01, 30.0)}
    window_samples = int(30 * fs)

    if len(raw) / fs > 30:
        iaf = []
        for start in tqdm(range(0, len(raw) - window_samples, window_samples)):
            freqs, psd = welch(raw[start : start + window_samples], fs=fs, nperseg=int(fs * 10))
            iaf.append(combine_simple(psd, freqs, config))
        return iaf

    # Short recording: use FOOOF on the whole signal
    freqs, psd = welch(raw, fs=fs, nperseg=int(fs * 10))
    fm = FOOOF(peak_width_limits=[0.1, 7.0], min_peak_height=0.001,
               peak_threshold=2., max_n_peaks=10, aperiodic_mode="fixed", verbose=False)
    try:
        fm.fit(freqs, psd)
    except Exception as e:
        print(f"FOOOF fitting error: {e}")
        return [np.nan]

    if fm.n_peaks_ == 0:
        return [np.nan]

    alpha_peaks = [p for p in fm.peak_params_ if 5 <= p[0] <= 18]
    return [max(alpha_peaks, key=lambda p: p[1])[0] if alpha_peaks else np.nan]
