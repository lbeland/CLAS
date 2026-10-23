"""
Variants of `approve_peak` for comparing peak-rejection strategies in
alpha_fast. Each factory function returns a function with the exact
signature iaf_compare.algorithms.alpha_fast expects to call:

    approve_peak(freqs, log_psd, psd_flat, psd_smooth, popt, gaussian,
                 aperiodic_simple, alpha_band, floor_value) -> bool

Variable units (see iaf_compare.algorithms.alpha_fast for derivation) --
everything is log10 scale, floor_value == 0.0 == "exactly on the aperiodic
fit":
    log_psd           log10(raw PSD), full freq_range
    aperiodic_simple  fitted aperiodic floor, log10 scale
    psd_flat          log_psd - aperiodic_simple -> log10 ratio, baseline 0.0
    psd_smooth        savgol-smoothed psd_flat -> log10 ratio, baseline 0.0
    gaussian          gaussian_peak(...) + floor_value -> same log10-ratio units as psd_flat
    popt              [amplitude, center_freq, std] of the fitted gaussian bump
"""
import numpy as np
from iaf_compare.algorithms import bic_peak_test_simple


def _snr_band_mask(freqs, popt):
    """Mask used for SNR integration: peak center +/- 2 std."""
    _, paf, std_gauss = popt
    return (freqs >= (paf - 2 * std_gauss)) & (freqs <= (paf + 2 * std_gauss))


# ---------------------------------------------------------------------------
# 1. BIC-only variants: same test, different input array for "residual"
# ---------------------------------------------------------------------------

def bic_test_raw(freqs, log_psd, aperiodic, gaussian, fmin, fmax):
    """BIC test against the raw (unwhitened, unsmoothed) log-power spectrum:
    null = aperiodic alone, alternative = aperiodic + gaussian bump."""
    fit_mask = (freqs >= fmin) & (freqs <= fmax)
    h0 = log_psd[fit_mask] - aperiodic[fit_mask]                         # null: no peak, just aperiodic
    h1 = log_psd[fit_mask] - (aperiodic[fit_mask] + gaussian[fit_mask])  # alternative: aperiodic + gaussian bump

    n = len(h1)
    if n < 4:
        return False, 0.0

    ss_h0 = np.sum(h0 ** 2)
    ss_h1 = np.sum(h1 ** 2)
    bic_h0 = n * np.log(max(ss_h0, np.nextafter(0, 1)) / n) + 2 * np.log(n)  # 2 params for the aperiodic fit (offset and exponent)
    bic_h1 = n * np.log(max(ss_h1, np.nextafter(0, 1)) / n) + 5 * np.log(n)  # 2 for aperiodic + 3 for Gaussian (amp, center, width)
    return (bic_h0 - bic_h1) > 0, bic_h0 - bic_h1


def make_bic_only(bic_input, range="alpha", floor_value=0.0):
    """bic_input: one of 'raw', 'psd_flat', 'psd_smooth'."""
    assert bic_input in ("raw", "psd_flat", "psd_smooth")
    assert range in ("full", "alpha", "peak")

    def approve_peak(freqs, log_psd, psd_flat, psd_smooth, popt, gaussian,
                      aperiodic_simple, alpha_band, floor_value=floor_value):
        _, paf, std_gauss = popt
        if range == "alpha":
            fmin, fmax = alpha_band[0], alpha_band[1]
        elif range == "peak":
            fmin, fmax = paf - std_gauss, paf + std_gauss
        elif range == "full":
            fmin, fmax = freqs[0], freqs[-1]

        if floor_value == "median":
            floor_value_use = np.median({"raw": log_psd, "psd_flat": psd_flat, "psd_smooth": psd_smooth}[bic_input])
        else:
            floor_value_use = floor_value

        if bic_input == "raw":
            peak_sig, delta_bic = bic_test_raw(freqs, log_psd, aperiodic_simple, gaussian, fmin, fmax)
        elif bic_input == "psd_flat":
            peak_sig, delta_bic = bic_peak_test_simple(freqs, psd_flat, gaussian, fmin, fmax, floor_value_use)
        elif bic_input == "psd_smooth":
            peak_sig, delta_bic = bic_peak_test_simple(freqs, psd_smooth, gaussian, fmin, fmax, floor_value_use)

        return bool(peak_sig)

    approve_peak.__name__ = f"bic_only__{bic_input}_{range}_fl{floor_value}"
    return approve_peak


# ---------------------------------------------------------------------------
# 2. SNR-only: no BIC gate at all -- just an SNR threshold, computed as the
#    integrated aperiodic-weighted excess power over the integrated aperiodic
#    (null) power, within +/- 2 std of the peak center.
# ---------------------------------------------------------------------------

def make_snr_only(snr_threshold=5.0):
    def approve_peak(freqs, log_psd, psd_flat, psd_smooth, popt, gaussian,
                      aperiodic_simple, alpha_band, floor_value=0.0):
        band_mask = _snr_band_mask(freqs, popt)
        eps = 1e-12

        # SNR = sum(aperiodic_lin * (10**gaussian - 1)) / sum(aperiodic_lin),
        # over |f - f0| <= 2*std 
        aperiodic_lin = np.power(10, aperiodic_simple)
        gauss_lin = np.power(10, gaussian) - 1.0
        signal_power = np.sum(aperiodic_lin[band_mask] * gauss_lin[band_mask])
        noise_power = np.sum(aperiodic_lin[band_mask])

        snr = signal_power / max(noise_power, eps)
        return bool(snr >= snr_threshold)

    approve_peak.__name__ = f"snr_only__thr{snr_threshold}"
    return approve_peak


# ---------------------------------------------------------------------------
# Registry: every strategy you want to compare, as (name -> approve_peak fn)
# ---------------------------------------------------------------------------

def build_strategy_registry(snr_thresholds=(3.0, 5.0, 8.0, 10.0, 15.0)):
    registry = {}

    # 1. BIC only -- different residual arrays x different fit ranges
    for bic_input in ("raw", "psd_smooth", "psd_flat"):
        for range in ("full", "alpha", "peak"):
            name = f"bic_only_{bic_input}_{range}"
            registry[name] = make_bic_only(bic_input, range=range)

    # 2. SNR only, at various thresholds
    for thr in snr_thresholds:
        name = f"snr_only__thr{thr}"
        registry[name] = make_snr_only(snr_threshold=thr)

    return registry
