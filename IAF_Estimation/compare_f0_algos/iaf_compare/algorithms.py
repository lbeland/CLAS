"""Peak-frequency (individual alpha frequency) estimators, plus the
``run_algorithms`` dispatch that runs each one on a single window.

``alpha_fast`` / ``alpha_fast_mt`` call ``approve_peak`` by bare name
so it resolves through this module's globals at call time -- that is what lets
``reject_compare`` swap in a different rejection strategy by rebinding
``iaf_compare.algorithms.approve_peak``. Keep it that way.
"""
import numpy as np
from fooof import FOOOF
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def _safe_log10(x):
    """log10 floored at the smallest positive float (smoothed PSD can dip
    slightly negative from Savitzky-Golay overshoot)."""
    return np.log10(np.maximum(x, np.nextafter(0, 1)))


def run_algorithms(window, psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt, config):
    return {
        "Maximum":      stupid_max(psd_welch, freq_bins_welch, config),
        "FOOOF":        fooof(psd_welch, freq_bins_welch, config),
        "RestingIAF":   philistine_iaf(psd, freq_bins, config),
        "alpha_fast":       alpha_fast(psd, freq_bins, config),
        "alpha_fast_mt":    alpha_fast_mt(psd_mt, freq_mt, config),
    }


def stupid_max(psd, freq_bins, config):
    band = (freq_bins >= config["alpha_band"][0]) & (freq_bins <= config["alpha_band"][1])
    psd_band = psd[band]
    max_bin = np.argmax(psd_band)
    freq_bins_band = freq_bins[band]
    est_pf = freq_bins_band[max_bin]
    # at the band's lower edge: reject if not above the bin just before it
    # (avoids picking low-freq noise when no clear peak is present)
    # if est_pf == freq_bins_band[0]:
    #     if psd_band[max_bin] < psd[freq_bins < config["alpha_band"][0]][-1]:
    #         est_pf = np.nan
    return est_pf


def parabolic_max(psd, freq_bins, config):
    band = (freq_bins >= config["alpha_band"][0]) & (freq_bins <= config["alpha_band"][1])
    psd_band = psd[band]
    max_bin = np.argmax(psd_band)
    freq_bins_band = freq_bins[band]
    est_pf = freq_bins_band[max_bin]
    # at the band's lower edge: reject if not above the bin just before it
    if est_pf == freq_bins_band[0]:
        if psd_band[max_bin] < psd[freq_bins < config["alpha_band"][0]][-1]:
            return np.nan

    # refine via parabolic interpolation if not at the band edges
    if 0 < max_bin < (psd_band.size - 1):
        y1, y2, y3 = psd_band[max_bin - 1], psd_band[max_bin], psd_band[max_bin + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            bin_hz = freq_bins_band[1] - freq_bins_band[0]
            return est_pf + delta * bin_hz

    return est_pf


def fooof(psd, freq_bins, config):
    """
    Parameterizing neural power spectra into periodic and aperiodic components
    10.1038/s41593-020-00744-x
    
    """
    fooof_model = FOOOF(peak_width_limits=[0.1, 4.0], min_peak_height=0.0,
                        peak_threshold=2., max_n_peaks=3, aperiodic_mode="fixed", verbose=False)
    try:
        fooof_model.fit(freq_bins, psd, config["freq_range"])
    except Exception as e:
        print(f"FOOOF fitting error: {e}")
        return np.nan

    if fooof_model.n_peaks_ == 0:
        return np.nan

    alpha_peaks = [p for p in fooof_model.peak_params_
                   if config["alpha_band"][0] <= p[0] <= config["alpha_band"][1]]
    return max(alpha_peaks, key=lambda p: p[1])[0] if alpha_peaks else np.nan


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


def alpha_fast_mt(psd, freq_bins, config):
    """Whitens the spectrum against a robust aperiodic fit, smooths it, and
    fits a Gaussian to the resulting peak to estimate the fundamental
    frequency. The whitened spectrum, its smoothed version, and the Gaussian
    all stay in log10-ratio space (0.0 = exactly on the aperiodic fit), which
    keeps amplitude/gaussian/floor_value on a consistent scale for the BIC
    test in approve_peak."""
    band = (freq_bins >= config["freq_range"][0]) & (freq_bins <= config["freq_range"][1])
    psd_band = psd[band]
    freqs = freq_bins[band]
    log_freqs = np.log10(freqs)

    resolution = freqs[1] - freqs[0]
    sav_gol_window_length = int(1.5 / resolution)
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
        return np.nan

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
        return np.nan
    popt = [amplitude, est_pf, std_gauss]

    gaussian = gaussian_peak(freqs, *popt) + floor_value

    peak_approved = approve_peak(freqs, log_psd, psd_flat, psd_smooth, popt, gaussian, aperiodic_simple, config["alpha_band"], floor_value=floor_value)

    if not peak_approved:
        return np.nan

    return est_pf


def alpha_fast(psd, freq_bins, config, return_diagnostics=False):
    """Whitens the spectrum against a robust aperiodic fit, smooths it, and
    fits a Gaussian to the resulting peak to estimate the fundamental
    frequency. The whitened spectrum, its smoothed version, and the Gaussian
    all stay in log10-ratio space (0.0 = exactly on the aperiodic fit), which
    keeps amplitude/gaussian/floor_value on a consistent scale for the BIC
    test in approve_peak.

    ``return_diagnostics=True`` additionally returns a dict of every
    intermediate array (both aperiodic fits, whitened/smoothed spectra,
    Gaussian fit, ...) for illustrating the procedure -- see
    iaf_compare.debug_plots.plot_alpha_fast_procedure. Doesn't change the
    normal (est_pf-only) return path at all."""
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

    def _diagnostics(popt=None, gaussian=None, peak_approved=None, std_gauss=None):
        return {
            "freqs": freqs, "psd": psd_band,
            "log_freqs": log_freqs, "log_psd": log_psd,
            "aperiodic_initial": aperiodic_initial,
            "aperiodic_refined": aperiodic_simple,
            "refit_mask": mask,
            "psd_flat": psd_flat,
            "psd_smooth": psd_smooth,
            "alpha_band": alpha_band,
            "est_pf": est_pf,
            "std_gauss": std_gauss,
            "floor_value": floor_value,
            "popt": popt,
            "gaussian": gaussian,
            "peak_approved": peak_approved,
        }

    # Reject outright if the smoothed peak doesn't clear the aperiodic fit
    # (log-ratio <= 0) -- there's no bump to fit a Gaussian to.
    if amplitude <= 0.0:
        if return_diagnostics:
            return np.nan, _diagnostics()
        return np.nan

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
        # print(std_gauss)
        if return_diagnostics:
            return np.nan, _diagnostics(std_gauss=std_gauss)
        return np.nan
    popt = [amplitude, est_pf, std_gauss]

    gaussian = gaussian_peak(freqs, *popt) + floor_value

    peak_approved = approve_peak(freqs, log_psd, psd_flat, psd_smooth, popt, gaussian, aperiodic_simple, config["alpha_band"], floor_value=floor_value)

    if return_diagnostics:
        return (est_pf if peak_approved else np.nan), _diagnostics(popt, gaussian, peak_approved, std_gauss)

    if not peak_approved:
        return np.nan

    return est_pf


def philistine_iaf(psd, freq_bins, config):
    """
    Toward a reliable, automated method of individual alpha frequency (IAF) quantification
    10.1111/psyp.13064

    """
    fmin, fmax = config["freq_range"]
    band = (freq_bins >= fmin) & (freq_bins <= fmax)
    freqs = freq_bins[band]
    psd_band = psd[band]
    resolution = freqs[1] - freqs[0]

    sav_gol_polyorder = 5
    # As recommended in Philistine paper
    sav_gol_window_length = int(2.5 / resolution)
    if sav_gol_window_length % 2 == 0:
        sav_gol_window_length += 1
    if sav_gol_polyorder >= sav_gol_window_length:
        sav_gol_window_length = sav_gol_polyorder + 2

    d0 = savgol_filter(psd_band, window_length=sav_gol_window_length, polyorder=sav_gol_polyorder)
    d1 = savgol_filter(psd_band, window_length=sav_gol_window_length, polyorder=sav_gol_polyorder,
                       deriv=1, delta=resolution)

    # --- noise floor: fitted log10(psd) + mpow prediction-error std devs ---
    mpow = config.get("mpow", 0.4)
    y = _safe_log10(psd_band)
    X = np.column_stack([freqs, np.ones_like(freqs)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma_hat = np.sqrt(np.sum(resid ** 2) / (len(y) - X.shape[1]))
    leverage = np.einsum('ij,jk,ik->i', X, np.linalg.inv(X.T @ X), X)
    min_pow = X @ beta + mpow * (sigma_hat * np.sqrt(1 + leverage))

    # --- dominant alpha peak via downward d1 zero-crossings within the band ---
    w = config["alpha_band"]
    n = len(freqs)
    lower_alpha = int(np.argmin(np.abs(freqs - w[0])))
    upper_alpha = int(np.argmin(np.abs(freqs - w[1])))
    lo, hi = max(lower_alpha - 1, 0), min(upper_alpha + 1, n - 2)
    idx = np.arange(lo, hi + 1)
    idx = idx[np.sign(d1[idx]) > np.sign(d1[idx + 1])]
    if len(idx) == 0:
        return np.nan
    maxima = np.where(d0[idx] >= d0[idx + 1], idx, idx + 1)

    top_bin = maxima[np.argmax(d0[maxima])]
    if _safe_log10(d0[top_bin]) <= min_pow[top_bin]:
        return np.nan  # doesn't clear the background noise floor

    # if a runner-up peak is within (1 - mdiff) of the top, restingIAF rejects
    # as "not clearly dominant" -- left disabled here, matching the original
    return freqs[top_bin]
