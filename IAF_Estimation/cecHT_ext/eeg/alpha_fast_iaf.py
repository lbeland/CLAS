"""alpha_fast IAF estimator -- ported from :func:`analysis.f0.alpha_fast`.

``gaussian_peak``, ``bic_peak_test_simple``, ``approve_peak`` and ``alpha_fast``
are a verbatim copy of ``analysis/f0.py`` (itself the production version of the
``alpha_fast`` estimator in ``IAF_Estimation/compare_f0_algos/iaf_compare/algorithms.py``,
plus a linear-scale SNR matching
``extensions/processors/FrequencyEstimation/FrequencyEstimation.cpp``). Kept as a
local copy rather than importing ``analysis.f0`` to avoid pulling in that
package's dependencies (mne-free here).

``alpha_fast_paf`` is the thin adapter the EEG pipeline calls: it builds the
same one-sided periodogram PSD that ``analysis.f0.estimate_f0`` feeds to
``alpha_fast`` and returns ``(peak_alpha_freq, snr)`` (``(nan, nan)`` if no peak
is approved).
"""

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter


# --- verbatim from analysis/f0.py -------------------------------------------------
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
    slope, intercept, _, _, _ = stats.linregress(log_freqs, log_psd)
    aperiodic_initial = log_freqs * slope + intercept  # log10-scale, first (unrefined) fit
    psd_flat = log_psd - aperiodic_initial  # log10 ratio: 0.0 = on fit

    # Select only samples that are exactly on (0.0) or below the perfect fit
    ratio_threshold = 0.0
    mask = psd_flat <= ratio_threshold
    freqs_refit = log_freqs[mask]
    psd_refit = log_psd[mask]
    # Re-fit using only the selected samples to ignore peak oscillations
    slope, intercept, _, _, _ = stats.linregress(freqs_refit, psd_refit)
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
# --- end -------------------------------------------------------------------------


def _periodogram(window, fs):
    """One-sided periodogram PSD, identical to analysis.f0.estimate_f0."""
    window = np.asarray(window, dtype=float)
    n = window.size
    freq_bins = np.fft.rfftfreq(n, 1 / fs)
    X = np.fft.rfft(window, n=n)
    psd = (np.abs(X) ** 2) / (fs * n)
    psd[1:-1] *= 2  # one-sided correction (except DC and Nyquist)
    return freq_bins, psd


def alpha_fast_paf(window, fs, alpha_band=(7.5, 14.0), freq_range=(0.1, 30.0)):
    """Peak alpha frequency and SNR of ``window`` via :func:`alpha_fast`.

    Parameters
    ----------
    window : 1-D array   raw signal samples for one IAF window
    fs : float           sampling rate (Hz)
    alpha_band : (float, float)   alpha search edges (Hz)
    freq_range : (float, float)   broadband range for the 1/f fit (Hz)

    Returns
    -------
    (paf_hz, snr) : (float, float)
        ``(nan, nan)`` if no peak is approved.
    """
    freq_bins, psd = _periodogram(window, fs)
    config = {"freq_range": tuple(freq_range), "alpha_band": tuple(alpha_band)}
    try:
        paf, _aperiodic_params, snr = alpha_fast(psd, freq_bins, config)
        return float(paf), float(snr)
    except Exception:  # noqa: BLE001 - match the pipeline's per-window robustness
        return np.nan, np.nan
