"""
Variants of `approve_peak` for comparing peak-rejection strategies in
combine_simple. Each factory function returns a function with the exact
signature IAF_tests.combine_simple expects to call:

    approve_peak(freqs, psd_safe, psd_flat, psd_smooth, popt, gaussian,
                 aperiodic_simple, alpha_band) -> bool

Variable units (see IAF_tests.combine_simple for derivation):
    psd_safe          raw PSD, linear power units, full freq_range
    aperiodic_simple  fitted aperiodic floor, LOG10 scale (so 10**aperiodic_simple
                       is the floor in linear units)
    psd_flat          psd_safe / 10**aperiodic_simple -> linear RATIO, baseline ~1.0
    psd_smooth        savgol-smoothed psd_flat -> linear RATIO, baseline ~1.0
    gaussian          gaussian_peak(...) + 1 -> same RATIO units/baseline as psd_flat
    popt              [amp, center_freq, std] of the fitted gaussian bump
"""
import numpy as np
from IAF_tests import bic_peak_test_simple


def _alpha_mask(freqs, alpha_band):
    fmin, fmax = alpha_band[0], alpha_band[1]
    return (freqs >= fmin) & (freqs <= fmax)


def _snr_band_mask(freqs, popt):
    """Mask used for SNR integration: peak center +/- 1 std (matches the
    style of the SNR test already sketched in IAF_reject_tests.approve_peak)."""
    _, paf, std_gauss = popt
    return (freqs >= (paf - std_gauss)) & (freqs <= (paf + std_gauss))


# ---------------------------------------------------------------------------
# 1. BIC-only variants: same test, different input array for "residual"
# ---------------------------------------------------------------------------
def bic_test(freqs, spectrum, aperiodic, gaussian, fmin, fmax):
    fit_mask = (freqs >= fmin) & (freqs <= fmax)
    h0 = spectrum[fit_mask] - np.power(10, aperiodic[fit_mask])  # null hypothesis: no peak, just aperiodic
    h1 = spectrum[fit_mask] - (gaussian[fit_mask] * np.power(10, aperiodic[fit_mask]))  # alternative: a Gaussian peak on top of the aperiodic

    n = len(h1)

    if n < 4:
        return False, 0.0

    ss_h0 = np.sum(h0 ** 2)
    ss_h1 = np.sum(h1 ** 2)
    bic_h0 = n * np.log(max(ss_h0, np.nextafter(0, 1)) / n) + 2 * np.log(n) # 2 params for the aperiodic fit (offset and exponent)
    bic_h1 = n * np.log(max(ss_h1, np.nextafter(0, 1)) / n) + 5 * np.log(n) # 2 for aperiodic + 3 for Gaussian (amp, center, width)
    return (bic_h0 - bic_h1) > 0, bic_h0 - bic_h1

def make_bic_only(bic_input, range="alpha", floor_value=1.0):
    """bic_input: one of 'psd_safe', 'psd_flat', 'psd_smooth'."""
    assert bic_input in ("orig", "psd_flat", "psd_smooth")
    assert range in ("full", "alpha", "peak")

    def approve_peak(freqs, psd_safe, psd_flat, psd_smooth, popt, gaussian,
                      aperiodic_simple, alpha_band, floor_value=1.0):
        _, paf, std_gauss = popt
        if range == "alpha":
            fmin, fmax = alpha_band[0], alpha_band[1]
        elif range == "peak":
            fmin, fmax = paf - std_gauss, paf + std_gauss
        elif range == "full":
            fmin, fmax = freqs[0], freqs[-1]

        if floor_value == "median":
            floor_value_use = np.median({"psd_safe": psd_safe, "psd_flat": psd_flat, "psd_smooth": psd_smooth}[bic_input])
        else:
            floor_value_use = floor_value

        if bic_input == "orig":
            peak_sig, delta_bic = bic_test(freqs, psd_safe, aperiodic_simple, gaussian, fmin, fmax)
        elif bic_input == "psd_flat":
            residual = psd_flat
            peak_sig, delta_bic = bic_peak_test_simple(freqs, residual, gaussian, fmin, fmax, floor_value_use)
        elif bic_input == "psd_smooth":
            residual = psd_smooth
            peak_sig, delta_bic = bic_peak_test_simple(freqs, residual, gaussian, fmin, fmax, floor_value_use)

        
        return bool(peak_sig)

    approve_peak.__name__ = f"bic_only__{bic_input}_{range}_fl{floor_value}"
    return approve_peak


# ---------------------------------------------------------------------------
# 2. BIC + SNR variants: always run the psd_smooth BIC test (matches the
#    current IAF_tests.py baseline) as a first gate, then ALSO require an
#    SNR threshold computed one of a few ways.
# ---------------------------------------------------------------------------

def make_bic_plus_snr(snr_mode, snr_threshold=5.0, bic_input="psd_smooth", floor_value=1.0):
    """
    snr_mode: one of
        'safe_minus_aperiodic'   S = psd_safe - 10**aperiodic_simple (linear PSD units)
                                  N = 10**aperiodic_simple
        'gauss_over_flat'        S = gaussian - 1, N = psd_flat - gaussian  (ratio units)
        'gauss_over_smooth'      S = gaussian - 1, N = psd_smooth - gaussian (ratio units)
        'gauss_over_smooth_abs'  like gauss_over_smooth, but N uses abs() so a
                                  smoothed dip below the gaussian doesn't make
                                  N negative and inflate/deflate the ratio
    """
    assert snr_mode in ("safe_minus_aperiodic", "gauss_over_flat",
                         "gauss_over_smooth", "gauss_over_smooth_abs")
    assert bic_input in ("psd_safe", "psd_flat", "psd_smooth")

    def approve_peak(freqs, psd_safe, psd_flat, psd_smooth, popt, gaussian,
                      aperiodic_simple, alpha_band, floor_value=1.0):
        fmin, fmax = alpha_band[0], alpha_band[1]
        residual = {"psd_safe": psd_safe, "psd_flat": psd_flat,
                    "psd_smooth": psd_smooth}[bic_input]
        
        if floor_value == "median":
            floor_value_use = np.median(residual)
        else:            
            floor_value_use = floor_value

        peak_sig, delta_bic = bic_peak_test_simple(freqs, residual, gaussian, fmin, fmax, floor_value_use)
        if not peak_sig:
            return False

        band_mask = _snr_band_mask(freqs, popt)
        eps = 1e-12

        if snr_mode == "safe_minus_aperiodic":
            aperiodic_lin = np.power(10, aperiodic_simple)
            P_signal = np.sum(psd_safe[band_mask] - aperiodic_lin[band_mask])
            P_noise  = np.sum(aperiodic_lin[band_mask])
        elif snr_mode == "gauss_over_flat":
            P_signal = np.sum(gaussian[band_mask] - floor_value_use)
            P_noise  = np.sum(psd_flat[band_mask] - gaussian[band_mask])
        elif snr_mode == "gauss_over_smooth":
            P_signal = np.sum(gaussian[band_mask] - floor_value_use)
            P_noise  = np.sum(psd_smooth[band_mask] - gaussian[band_mask])
        elif snr_mode == "gauss_over_smooth_abs":
            P_signal = np.sum(gaussian[band_mask] - floor_value_use)
            P_noise  = np.sum(np.abs(psd_smooth[band_mask] - gaussian[band_mask]))

        snr = P_signal / max(P_noise, eps)
        return bool(snr >= snr_threshold)

    approve_peak.__name__ = f"bic_{bic_input}_plus_snr__{snr_mode}__thr{snr_threshold}_fl{floor_value}"
    return approve_peak

# ---------------------------------------------------------------------------
# 3. SNR-only variants: same SNR modes as above, but skip the BIC test
# ---------------------------------------------------------------------------
def make_snr_only(snr_mode, snr_threshold=5.0, floor_value=1.0):
    assert snr_mode in ("model_minus_aperiodic", "orig_minus_aperiodic", "gauss_over_flat",
                         "gauss_over_smooth", "gauss_over_smooth_abs")

    def approve_peak(freqs, psd_safe, psd_flat, psd_smooth, popt, gaussian,
                      aperiodic_simple, alpha_band, floor_value=1.0):
        band_mask = _snr_band_mask(freqs, popt)
        eps = 1e-12

        if snr_mode == "orig_minus_aperiodic":
            aperiodic_lin = np.power(10, aperiodic_simple)
            P_signal = np.sum(psd_safe[band_mask] - aperiodic_lin[band_mask])
            P_noise  = np.sum(aperiodic_lin[band_mask])
        elif snr_mode == "model_minus_aperiodic":
            aperiodic_lin = np.power(10, aperiodic_simple)
            P_signal = np.sum((aperiodic_lin[band_mask] * gaussian[band_mask]) - aperiodic_lin[band_mask])
            P_noise  = np.sum(aperiodic_lin[band_mask])
        elif snr_mode == "gauss_over_flat":
            floor_value_use = floor_value if floor_value != "median" else np.median(psd_flat)
            P_signal = np.sum(gaussian[band_mask] - floor_value_use)
            P_noise  = np.sum(psd_flat[band_mask] - gaussian[band_mask])
        elif snr_mode == "gauss_over_smooth":
            floor_value_use = floor_value if floor_value != "median" else np.median(psd_smooth)
            P_signal = np.sum(gaussian[band_mask] - floor_value_use)
            P_noise  = np.sum(psd_smooth[band_mask] - gaussian[band_mask])
        elif snr_mode == "gauss_over_smooth_abs":
            floor_value_use = floor_value if floor_value != "median" else np.median(psd_smooth)
            P_signal = np.sum(gaussian[band_mask] - floor_value_use)
            P_noise  = np.sum(np.abs(psd_smooth[band_mask] - gaussian[band_mask]))

        snr = P_signal / max(P_noise, eps)
        return bool(snr >= snr_threshold)

    approve_peak.__name__ = f"snr_only__{snr_mode}__thr{snr_threshold}__fl{floor_value}"
    return approve_peak


# ---------------------------------------------------------------------------
# Registry: every strategy you want to compare, as (name -> approve_peak fn)
# ---------------------------------------------------------------------------

def build_strategy_registry(snr_thresholds=(3.0, 5.0, 8.0, 10.0, 15.0)):
    registry = {}

    # 1.x BIC-only
    for bic_input in ("orig","psd_smooth", "psd_flat"):
        for range in ("full", "alpha", "peak"):
            if bic_input == "orig":
                name = f"bic_only_{bic_input}_{range}"
                registry[name] = make_bic_only(bic_input, range=range)
            else:
                # for floor_value in (1.0, "median"):
                name = f"bic_only_{bic_input}_{range}" #_fl{floor_value}"
                registry[name] = make_bic_only(bic_input, range=range, floor_value=1.0)

    # # 2.x BIC + SNR, swept over a couple of thresholds so you can see the
    # # sensitivity/specificity tradeoff move, not just one operating point
    # for snr_mode in ("orig_minus_aperiodic", "gauss_over_flat",
    #                   "gauss_over_smooth"):
    #     for thr in snr_thresholds:
    #         name = f"bic_smooth_plus_snr_{snr_mode}_thr{thr}"
    #         registry[name] = make_bic_plus_snr(snr_mode, snr_threshold=thr)

    # 3. SNR only
    for snr_mode in ("model_minus_aperiodic", "orig_minus_aperiodic", "gauss_over_flat",
                     "gauss_over_smooth"):
        for thr in snr_thresholds:
            if snr_mode in ("model_minus_aperiodic", "orig_minus_aperiodic"):
                name = f"snr_only_{snr_mode}__thr{thr}"
                registry[name] = make_snr_only(snr_mode, snr_threshold=thr)
            else:
                # for floor_value in (1.0, "median"):
                name = f"snr_{snr_mode}_thr{thr}" #_fl{floor_value}"
                registry[name] = make_snr_only(snr_mode, snr_threshold=thr, floor_value=1.0)

    return registry
