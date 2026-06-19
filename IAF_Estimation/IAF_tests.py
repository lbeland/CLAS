import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fooof import FOOOF
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from neurodsp.sim import sim_bursty_oscillation
from pathlib import Path
from tqdm import tqdm
import h5py
from scipy.signal import welch
from multiprocessing import Pool

BASE_FOLDER = Path(__file__).parent

# ---------------------------------------------------------------------------
# Signal generation — principled amplitude model
# ---------------------------------------------------------------------------
#
#   aperiodic_ref_power : float
#       PSD of the aperiodic component at f_rotation. Acts as the global
#       power anchor. Units are arbitrary (e.g. µV²/Hz normalised to 1.0).
#
#   f_rotation : float
#       Spectral pivot frequency for the aperiodic shape (default: 1.0 Hz).
#       The aperiodic PSD equals aperiodic_ref_power exactly at this frequency.
#       Keep this fixed (e.g. 1 Hz) and use aperiodic_ref_freq for SNR.
#
#   aperiodic_ref_freq : float
#       The frequency at which the aperiodic floor is measured when computing
#       peak_snr_db. Decoupled from f_rotation so you can ask e.g.:
#         "how far is the peak at 10 Hz above the aperiodic floor at 5 Hz?"
#       Set to None to fall back to carrier_freq (old behaviour).
#
#   peak_snr_db : float
#       Peak power relative to the PSD at aperiodic_ref_freq.
#         +10 dB → peak clearly above background at ref freq
#           0 dB → peak power equals aperiodic floor at ref freq
#          -6 dB → peak buried below aperiodic at ref freq → stupid_max fails
#
#       The actual aperiodic PSD at aperiodic_ref_freq is derived from the
#       shape: S_ap(f_ref) = ref_power * (f_ref / f_rotation)^(-β)
#       so peak_power = S_ap(f_ref) * 10^(peak_snr_db / 10)
#
#   noise_snr_db : float
#       Noise PSD relative to the peak power.
#
# ---------------------------------------------------------------------------

def main():
    fixed = {
        "fs":                   10000.0,
        "signal_length_sec":    30,
        "freq_range":           (0.01, 30.0),
        "alpha_band":           (5, 18),
        "pink_ax_r2":           0.9,
        # Aperiodic anchor power at f_rotation (arbitrary units, ~1 µV²/Hz)
        "aperiodic_ref_power":  1.0,
        # Spectral pivot — fix at 1 Hz so the shape is independent of carrier
        "f_rotation":           1.0,
        # Frequency at which the aperiodic floor is measured for peak_snr_db.
        # None → use carrier_freq (old behaviour).
        # Example: carrier_freq=10, aperiodic_ref_freq=5 → peak_snr_db is
        # measured vs. aperiodic at 5 Hz, so negative values make stupid_max
        # pick 5 Hz instead of 10 Hz.
        "aperiodic_ref_freq":   5.0,
    }

    default = {
        "carrier_freq":         14.0,
        # Peak shape in frequency domain
        "carrier_waveform":     "gaussian",   # "gaussian" | "sine" | "burst"
        # Aperiodic component
        "aperiodic_exponent":   2.0,          # β — slope of 1/f^β
        "has_aperiodic":        True,
        # Peak(s)
        "n_peaks":              1,
        "peak_bw":              0.5,          # Gaussian σ in Hz
        # KEY PARAM: peak power relative to aperiodic floor at aperiodic_ref_freq
        "peak_snr_db":          10.0,
        # Noise
        "noise_type":           "white",       # "None" | "white" | "pink"
        # Noise PSD relative to power at carrier_freq
        "noise_snr_db":         -20.0,
        # Analysis
        "window_length_sec":    10,
    }

    cf = [value + 0.1*idx for idx, value in enumerate(np.arange(6, 16))]

    sweeps = {
        "window_length_sec":    [5, 10, 20],
        "peak_snr_db":          [20, 10, 3, 0, -3],
        "peak_bw":              [0.1, 0.5, 1.0, 2.0],
        "aperiodic_exponent":   [1, 2, 3],
        "has_aperiodic":        [True, False],
        "carrier_freq":         cf,
        # "noise_type":           ["None", "white"],
        "carrier_waveform":     ["gaussian", "sine", "burst"],
        "n_peaks":              [1, 2, 3],
    }

    # Sweeps for 0 peak default
    # sweeps = {
    #     "window_length_sec":    [5, 10, 20],
    #     "aperiodic_exponent":   [1, 2, 3],
    #     "has_aperiodic":        [True, False],
    # }

    seen = set()
    conditions = []

    for param, values in sweeps.items():
        sweep_default = {**default}

        for val in values:
            config = {**fixed, **sweep_default, param: val}
            key = tuple(sorted(config.items(), key=lambda x: str(x)))
            if key not in seen:
                seen.add(key)
                conditions.append(config)

    print(f"Total conditions: {len(conditions)}")

    N_SEEDS = 5  # start small; increase once runtime is acceptable

    conditions_with_seeds = [
        (i * N_SEEDS + seed, cfg, seed)
        for i, cfg in enumerate(conditions)
        for seed in range(N_SEEDS)
    ]

    n_workers = 5 # max(1, cpu_count() - 1)
    with Pool(n_workers) as pool:
        all_results = list(tqdm(
            pool.imap(process_condition, conditions_with_seeds),
            total=len(conditions_with_seeds),
            desc="Processing conditions",
        ))

    with h5py.File(BASE_FOLDER / "iaf_results.h5", "w") as hf:
        hf.attrs["created"] = str(pd.Timestamp.now())
        conds_grp = hf.create_group("conditions")

        for cond_idx, config, gt, algo_results, (first_freq_spectrum, first_window) in tqdm(all_results, desc="Writing HDF5"):
            grp = conds_grp.create_group(f"{cond_idx:04d}")

            for k, v in config.items():
                if isinstance(v, tuple):
                    grp.attrs[k] = list(v)
                else:
                    grp.attrs[k] = v if v is not None else "none"

            freq_bins, psd = first_freq_spectrum
            grp.create_dataset("spectrum_psd",       data=psd,          compression="gzip")
            grp.create_dataset("spectrum_freq_bins", data=freq_bins,    compression="gzip")
            grp.create_dataset("first_window",       data=first_window, compression="gzip")
            grp.create_dataset("ground_truth",       data=gt,           compression="gzip")

            for algo, data in algo_results.items():
                algo_grp = grp.create_group(algo)
                algo_grp.create_dataset("estimates", data=data["estimates"], compression="gzip")
                algo_grp.create_dataset("errors",    data=data["errors"],    compression="gzip")
                for metric in ("mae", "rmse", "std", "fn", "fp", "n", "fail_rate", "accuracy", "f1_score"):
                    algo_grp.attrs[metric] = data[metric]


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_aperiodic(n_samples, fs, exponent, f_rotation, ref_power, rng):
    """
    Generate a 1/f^β aperiodic signal whose PSD at f_rotation equals ref_power.

    Parameters
    ----------
    n_samples   : int
    fs          : float
    exponent    : float   β — spectral slope
    f_rotation  : float   pivot frequency in Hz
    ref_power   : float   desired PSD at f_rotation (linear power units)

    Returns
    -------
    signal : np.ndarray  (n_samples,)
    """
    x = rng.standard_normal(n_samples)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)

    # Save DC bin and skip it (undefined for 1/f)
    X_dc = X[0]
    freqs_ndc = freqs[1:]
    X_ndc = X[1:]

    # Amplitude scaling: S(f) ∝ f^-β  →  amplitude ∝ f^(-β/2)
    scaling = (freqs_ndc / f_rotation) ** (-exponent / 2)
    X_ndc = X_ndc * scaling

    X_scaled = np.concatenate([[X_dc], X_ndc])
    sig = np.fft.irfft(X_scaled, n=n_samples)

    # Normalise so PSD at f_rotation equals ref_power.
    # PSD bin at f_rotation = |X[k]|² / (fs * n_samples)
    f_rot_bin = np.argmin(np.abs(freqs_ndc - f_rotation))
    current_psd_at_rot = (np.abs(X_ndc[f_rot_bin]) ** 2) / (fs * n_samples)
    amplitude_scale = np.sqrt(ref_power / max(current_psd_at_rot, np.nextafter(0, 1)))
    return sig * amplitude_scale


def generate_peak(n_samples, fs, center_freq, bw, peak_psd, rng):
    """
    Generate a narrowband oscillation with a Gaussian spectrum.
    Scaled so that PSD at center_freq equals peak_psd — consistent with
    how peak_snr_db is defined (i.e. PSD at the peak tip, not integrated power).
 
    Parameters
    ----------
    n_samples   : int
    fs          : float
    center_freq : float   Hz
    bw          : float   Gaussian σ in Hz
    peak_psd    : float   desired PSD at center_freq (linear units, same scale
                          as aperiodic PSD at aperiodic_ref_freq)
 
    Returns
    -------
    signal : np.ndarray  (n_samples,)
    """
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
 
    # Gaussian amplitude envelope (unit peak at center_freq)
    amp_envelope = np.exp(-0.5 * ((freqs - center_freq) / bw) ** 2)
 
    # Current PSD at center bin = |amp_envelope[k]|² / (fs * n_samples)
    # At center: amp_envelope = 1.0, so current_psd_at_center = 1 / (fs * n_samples)
    # Scale so that PSD at center equals peak_psd
    target_amp_at_center = np.sqrt(peak_psd * fs * n_samples)
    amp_envelope *= target_amp_at_center   # peak of envelope hits target, shoulders scale with it
 
    phases = rng.uniform(0, 2 * np.pi, len(amp_envelope))
    X = amp_envelope * np.exp(1j * phases)
 
    sig = np.fft.irfft(X, n=n_samples)
    return sig


def generate_sine_peak(n_samples, fs, center_freq, peak_power, t):
    """
    Generate a pure sine with the given total power.
    """
    sig = np.sin(2 * np.pi * center_freq * t)

    # Scale to desired power: power of cos = 0.5, so RMS² = 0.5
    target_total_power = peak_power * fs / n_samples
    current_power = np.mean(sig ** 2)
    sig *= np.sqrt(target_total_power / max(current_power,np.nextafter(0, 1)))
    return sig


def generate_burst_peak(n_samples, fs, center_freq, peak_power,
                        n_cycles_on=10, n_cycles_off=30):
    """Bursty oscillation with desired average power."""

    sig = sim_bursty_oscillation(n_samples/fs,fs, center_freq, burst_def="durations",
                                burst_params={"n_cycles_burst": n_cycles_on, "n_cycles_off": n_cycles_off})

    current_power = np.mean(sig ** 2)
    sig *= np.sqrt(peak_power / max(current_power, np.nextafter(0, 1)))
    return sig


def generate_white_noise(n_samples, fs, noise_psd, rng):
    """
    White noise with flat PSD = noise_psd (linear units per Hz).
    Total power = noise_psd * (fs/2).
    """

    noise = rng.standard_normal(n_samples)
    # White noise variance = noise_psd * (fs/2)
    target_std = np.sqrt(noise_psd * fs / 2)
    noise = noise / max(np.std(noise), np.nextafter(0, 1)) * target_std
    return noise


def generate_signal(config, rng):
    fs       = config["fs"]
    n        = int(config["signal_length_sec"] * fs)
    t        = np.arange(n) / fs
    inst_freqs = np.full(n, np.nan)

    ref_power  = config["aperiodic_ref_power"]   # PSD at reference point
    f_rot      = config["f_rotation"]             # spectral pivot (fixed, e.g. 1 Hz)
    beta       = config["aperiodic_exponent"]

    # Aperiodic PSD at the reference frequency used for SNR comparisons.
    # S_ap(f_ref) = ref_power * (f_ref / f_rotation)^(-β)
    f_ref = config["aperiodic_ref_freq"]


    # ------------------------------------------------------------------
    # 1. Aperiodic component
    # ------------------------------------------------------------------
    signal = np.zeros(n)

    if config["has_aperiodic"]:
        signal += generate_aperiodic(
            n, fs,
            exponent=beta,
            f_rotation=f_rot,
            ref_power=ref_power,
            rng=rng
        )
        ap_psd_at_ref = ref_power * (f_ref / f_rot) ** (-beta)
    else:
        ap_psd_at_ref = ref_power

    # ------------------------------------------------------------------
    # 2. Peak(s)
    # ------------------------------------------------------------------
    # Peak power is defined relative to the aperiodic PSD at aperiodic_ref_freq,
    # NOT at the carrier frequency. This lets you ask:
    #   "is my 10 Hz peak above or below the aperiodic floor at 5 Hz?"
    # peak_snr_db < 0  →  peak weaker than aperiodic at ref freq  →  stupid_max fails
    peak_power = ap_psd_at_ref * (10 ** (config["peak_snr_db"] / 10))

    if config["n_peaks"] > 0:
        waveform = config["carrier_waveform"]
        cf       = config["carrier_freq"]
        inst_freqs[:] = cf

        if waveform == "sine":
            sig_peak = generate_sine_peak(n, fs, cf, peak_power, t)
        elif waveform == "burst":
            sig_peak = generate_burst_peak(n, fs, cf, peak_power)
        else:  # gaussian (default)
            sig_peak = generate_peak(n, fs, cf, config["peak_bw"], peak_power, rng)

        signal += sig_peak

        # Additional peaks at random frequencies within the alpha band
        for idx in range(config["n_peaks"] - 1):
            extra_freq = cf + (-1)**idx * 2  # Alternate above and below the main peak with 2 Hz spacing
            # Extra peaks are half the power of the main peak
            extra_power = peak_power * 0.2
            signal += generate_peak(n, fs, extra_freq, config["peak_bw"], extra_power, rng)

    # ------------------------------------------------------------------
    # 3. Noise
    # ------------------------------------------------------------------
    if config["noise_type"] != "None":
        # noise_snr_db is noise PSD relative to peak power
        noise_psd = peak_power * (10 ** (config["noise_snr_db"] / 10))

        if config["noise_type"] == "white":
            signal += generate_white_noise(n, fs, noise_psd, rng)
        elif config["noise_type"] == "pink":
            # Pink noise: use exponent=1, same pivot, noise_psd sets level
            signal += generate_aperiodic(n, fs, exponent=1.0,
                                         f_rotation=f_rot,
                                         ref_power=noise_psd, rng=rng)

    return signal, inst_freqs


# ---------------------------------------------------------------------------
# Pipeline (unchanged except calling new generate_signal)
# ---------------------------------------------------------------------------

def process_condition(args):
    cond_idx, config, seed = args
    # print(config)    
    try:
        rng = np.random.default_rng(seed)
        signal, inst_freq = generate_signal(config, rng)
        estimates_per_algo, ground_truth, (first_freq_spectrum, first_window) = \
            run_window_analysis(signal, inst_freq, config)

        algo_results = {}
        for algo, estimates in estimates_per_algo.items():
            estimates = np.asarray(estimates, dtype=float)
            errors = []
            for est, gt in zip(estimates, ground_truth):
                if ~np.isnan(est) and np.isnan(gt):
                    errors.append(est)
                elif ~np.isnan(est) and ~np.isnan(gt):
                    errors.append(est - gt)
            algo_results[algo] = {
                "estimates": estimates,
                "errors":    np.array(errors),
                **compute_metrics(estimates, ground_truth),
            }
        return cond_idx, config, ground_truth, algo_results, (first_freq_spectrum, first_window)
    except Exception as e:
        print(f"Error processing condition\n{config}\n: {e}")
        raise


def run_window_analysis(signal, inst_freq, config):
    ground_true_freqs = []
    signal_length = len(signal)
    results_all = {}
    window_length = int(config["window_length_sec"] * config["fs"])
    first_freq_spectrum = None
    first_window = None

    step = max(window_length // 5, int(config["fs"]))  # at most 1s step
    for idx in range(window_length, signal_length, step):
        window = signal[idx - window_length:idx]

        nperseg = int(min(window_length, 7 * config["fs"]))
        freq_bins_welch, psd_welch = welch(window, fs=config["fs"], nperseg=nperseg, noverlap=nperseg//1.5)

        freq_bins = np.fft.rfftfreq(window_length, 1 / config["fs"])
        X = np.fft.rfft(window, n=window_length)
        # Periodogram PSD: |X|^2 / (fs * N) — matches Welch units (power per Hz)
        psd = (np.abs(X) ** 2) / (config["fs"] * window_length)
        psd[1:-1] *= 2  # Correct for dropping negative freqs in one-sided spectrum (except DC and Nyquist)

        if first_freq_spectrum is None:
            first_freq_spectrum = (freq_bins, psd)
            first_window = window

        ground_true_freqs.append(inst_freq[idx - 1])

        for algo, estimate in run_algorithms(psd, psd_welch, freq_bins, freq_bins_welch, config).items():
            if algo not in results_all:
                results_all[algo] = []
            results_all[algo].append(estimate)

    return results_all, np.array(ground_true_freqs), (first_freq_spectrum, first_window)


def compute_metrics(estimates, ground_truth):
    estimates    = np.asarray(estimates,    dtype=object)
    ground_truth = np.asarray(ground_truth, dtype=object)

    n = len(estimates)
    abs_errors = []
    fp, fn, tn, tp = 0, 0, 0, 0

    for est, gt in zip(estimates, ground_truth):
        est_none = est is None or (isinstance(est, float) and np.isnan(est))
        gt_none  = gt  is None or (isinstance(gt,  float) and np.isnan(gt))

        if gt_none and est_none:
            tn += 1
        elif gt_none and not est_none:
            fp += 1
            abs_errors.append(est)
        elif not gt_none and est_none:
            fn += 1
        else:
            tp += 1
            abs_errors.append(abs(est - gt))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    accuracy = (tp + tn) / n if n > 0 else np.nan
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

    return {
        "mae":            np.mean(abs_errors) if abs_errors else np.nan,
        "rmse":           np.sqrt(np.mean(np.array(abs_errors) ** 2)) if abs_errors else np.nan,
        "std":            np.std(abs_errors) if abs_errors else np.nan,
        "fn":             fn,
        "fp":             fp,
        "n":              n,
        "fail_rate":      (fn+fp) / n,
        "accuracy":       accuracy,
        "f1_score":       f1_score,
    }


# ---------------------------------------------------------------------------
# Algorithms (unchanged)
# ---------------------------------------------------------------------------

def run_algorithms(psd, psd_welch, freq_bins, freq_bins_welch, config):
    return {
        "stupid_max":   stupid_max(psd, freq_bins, config),
        "fooof":        fooof(psd_welch, freq_bins_welch, config),
        "philistine":   philistine_iaf(psd, freq_bins, config),
        "combine_complex":      combine_algo(psd, freq_bins, config),
        "combine_simple":       combine_simple(psd, freq_bins, config),
    }


def stupid_max(psd, freq_bins, config):
    band = (freq_bins >= config["alpha_band"][0]) & (freq_bins <= config["alpha_band"][1])
    psd_band = psd[band]
    max_bin = np.argmax(psd_band)
    freq_bins_band = freq_bins[band]
    paf = freq_bins_band[max_bin]
    # if paf at the lower edge of band, check if power is higher than bin right before the band
    # to avoid picking low-freq noise when no clear peak is present
    if paf == freq_bins_band[0]:
        if psd_band[max_bin] < psd[freq_bins < config["alpha_band"][0]][-1]:
            paf = np.nan
    return paf


def parabolic_max(psd, freq_bins, config):
    band = (freq_bins >= config["alpha_band"][0]) & (freq_bins <= config["alpha_band"][1])
    psd_band = psd[band]
    max_bin = np.argmax(psd_band)
    freq_bins_band = freq_bins[band]
    paf = freq_bins_band[max_bin]
    # check if max power is higher than bin right before the band (to avoid picking low-freq noise when no clear peak is present)
    # if paf at the lower edge of band, check if power is higher than bin right before the band
    # to avoid picking low-freq noise when no clear peak is present
    if paf == freq_bins_band[0]:
        if psd_band[max_bin] < psd[freq_bins < config["alpha_band"][0]][-1]:
            return np.nan

    # if not at the edges, do parabolic interpolation to refine the peak estimate
    if 0 < max_bin < (psd_band.size - 1):
        y1, y2, y3 = psd_band[max_bin - 1], psd_band[max_bin], psd_band[max_bin + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            bin_hz = freq_bins_band[1] - freq_bins_band[0]
            return paf + delta * bin_hz

    return paf


def fooof(psd, freq_bins, config):
    fm = FOOOF(peak_width_limits=[0.1, 4.0], min_peak_height=0.0,
               peak_threshold=2., max_n_peaks=3, aperiodic_mode="fixed", verbose=False)
    try:
        fm.fit(freq_bins, psd, config["freq_range"])
    except Exception as e:
        print(f"FOOOF fitting error: {e}")
        return np.nan

    if fm.n_peaks_ == 0:
        return np.nan

    alpha_peaks = [p for p in fm.peak_params_
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
    bic_h1 = n * np.log(max(ss_h1, np.nextafter(0, 1)) / n) + 3 * np.log(n) # 3 params for Gaussian (amp, center, width)
    return (bic_h0 - bic_h1) > 0, bic_h0 - bic_h1

def bic_peak_test(residual, freqs, fmin, fmax):
    fit_mask = (freqs >= fmin) & (freqs <= fmax)
    fit_freqs = freqs[fit_mask]
    fit_resid = residual[fit_mask]
    n = len(fit_resid)

    if n < 4:
        return False, 0.0

    ss_h0 = np.sum(fit_resid ** 2)
    peak_idx = np.argmax(fit_resid)
    amp_guess = max(fit_resid[peak_idx], 0.01)
    center_guess = fit_freqs[peak_idx]
    width_guess = (fmax - fmin) / 4

    popt, _ = curve_fit(
        gaussian_peak, fit_freqs, fit_resid,
        p0=[amp_guess, center_guess, width_guess],
        bounds=([0.0, fmin, 0.25], [np.inf, fmax, fmax - fmin]),
        maxfev=10_000,
    )

    ss_h1 = np.sum((fit_resid - gaussian_peak(fit_freqs, *popt)) ** 2)
    bic_h0 = n * np.log(ss_h0 / n)
    bic_h1 = n * np.log(max(ss_h1, np.nextafter(0, 1)) / n) + 3 * np.log(n)
    return (bic_h0 - bic_h1) > 0, bic_h0 - bic_h1


def combine_algo(psd, freq_bins, config):
    band = (freq_bins >= config["freq_range"][0]) & (freq_bins <= config["freq_range"][1])
    psd_band = psd[band]
    freqs = freq_bins[band]

    resolution = freqs[1] - freqs[0]
    sav_gol_window_length = int(2.5 / resolution)
    if sav_gol_window_length % 2 == 0:
        sav_gol_window_length += 1

    sav_gol_polyorder = 3
    if sav_gol_polyorder >= sav_gol_window_length:
        sav_gol_window_length = sav_gol_polyorder + 2

    fm = FOOOF(peak_width_limits=(0.1, 4.0), max_n_peaks=3, min_peak_height=0.0,
               peak_threshold=2.0, aperiodic_mode="fixed", verbose=False)
    try:
        fm.fit(freqs, psd_band, config["freq_range"])
    except Exception as e:
        print(f"FOOOF fitting error: {e}")
        return np.nan

    offset, exponent = fm.aperiodic_params_
    aperiodic = offset - exponent * np.log10(freqs)

    psd_safe = np.maximum(psd_band, np.nextafter(0, 1))
    residual = np.log10(psd_safe) - aperiodic
    psd_flat = np.power(10, residual)
    psd_flat = np.clip(psd_flat, np.nextafter(0, 1), None)

    if not np.all(np.isfinite(psd_flat)):
        return np.nan

    psd_smooth = savgol_filter(psd_flat, window_length=sav_gol_window_length, polyorder=sav_gol_polyorder)
    eps = np.nextafter(0, 1)  # Smallest positive float
    _, _, r, _, _ = stats.linregress(np.log(freqs), np.log(np.maximum(psd_smooth, eps)))

    fmin, fmax = config["alpha_band"][0], config["alpha_band"][1]
    peak_sig, delta_bic = bic_peak_test(residual, freqs, fmin, fmax)
    alpha_band = (freqs >= fmin) & (freqs <= fmax)

    if r ** 2 > config["pink_ax_r2"] or not peak_sig:
        return np.nan

    max_bin = np.argmax(psd_smooth[alpha_band])
    paf = freqs[alpha_band][max_bin]
    alpha_weights = psd_smooth[alpha_band]
    cog = (float(np.average(freqs[alpha_band], weights=alpha_weights))
           if np.any(alpha_weights > 0) else None)
    if cog is None:
        return np.nan

    if 0 < max_bin < (psd_smooth[alpha_band].size - 1):
        y1, y2, y3 = psd_smooth[alpha_band][max_bin - 1], psd_smooth[alpha_band][max_bin], psd_smooth[alpha_band][max_bin + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            return paf + delta * resolution
        
    return paf


def approve_peak(freqs, psd_safe, psd_flat, psd_smooth, popt, gaussian, aperiodic_simple, alpha_band, floor_value):
    fmin, fmax = freqs[0], freqs[-1]
    peak_sig, delta_bic = bic_peak_test_simple(freqs, psd_flat, gaussian, fmin, fmax, floor_value)

    # _, paf, std_gauss = popt
    # band_mask = (freqs >= (paf - std_gauss)) & (freqs <= (paf + std_gauss))
    # aperiodic_lin = np.power(10, aperiodic_simple)
    # P_signal = np.sum(psd_safe[band_mask] - aperiodic_lin[band_mask])
    # P_noise  = np.sum(aperiodic_lin[band_mask])

    # snr = P_signal / max(P_noise, 1e-12)
    # peak_sig = bool(snr >= 15)

    if not peak_sig:
        return False
    else:
        return True

def combine_simple(psd, freq_bins, config):
    band = (freq_bins >= config["freq_range"][0]) & (freq_bins <= config["freq_range"][1])
    psd_band = psd[band]
    freqs = freq_bins[band]

    resolution = freqs[1] - freqs[0]
    sav_gol_window_length = int(2.5 / resolution)
    if sav_gol_window_length % 2 == 0:
        sav_gol_window_length += 1

    sav_gol_polyorder = 3
    if sav_gol_polyorder >= sav_gol_window_length:
        sav_gol_window_length = sav_gol_polyorder + 2

    eps = np.nextafter(0, 1)  # Smallest positive float
    psd_safe = np.maximum(psd_band, eps)
    slope, intercept, _, _, _ = stats.linregress(np.log10(freqs), np.log10(psd_safe))
    aperiodic_simple = np.log10(freqs) * slope + intercept  # log10-scale
    psd_flat = psd_safe / np.power(10, aperiodic_simple)  # ratio: 1.0 = on fit

    # Select only samples that are exactly on (1) or below the perfect fit
    ratio_threshold = 1.0
    mask = psd_flat <= ratio_threshold
    freqs_refit = freqs[mask]
    psd_refit = psd_safe[mask]
    # Re-fit using only the selected samples to ignore peak oscillations
    slope, intercept, _, _, _ = stats.linregress(np.log10(freqs_refit), np.log10(psd_refit))
    aperiodic_simple = np.log10(freqs) * slope + intercept  # log10-scale
    psd_flat = psd_safe / np.power(10, aperiodic_simple)

    psd_smooth = savgol_filter(psd_flat, window_length=sav_gol_window_length, polyorder=sav_gol_polyorder)

    fmin, fmax = config["alpha_band"][0], config["alpha_band"][1]

    alpha_band = (freqs >= fmin) & (freqs <= fmax)

    psd_alpha_band = psd_smooth[alpha_band]
    fit_freqs = freqs[alpha_band]

    max_bin = np.argmax(psd_alpha_band)
    paf = freqs[alpha_band][max_bin]

    if 0 < max_bin < (psd_alpha_band.size - 1):
        y1, y2, y3 = psd_alpha_band[max_bin - 1], psd_alpha_band[max_bin], psd_alpha_band[max_bin + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            paf += delta * resolution


    floor_value = 1

    amp_guess = psd_smooth[alpha_band][max_bin] - floor_value
    center_guess = fit_freqs[max_bin]

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
        return np.nan
    popt = [amp_guess, center_guess, std_gauss]

    gaussian = gaussian_peak(freqs, *popt) + floor_value
    
    peak_approved = approve_peak(freqs, psd_safe, psd_flat, psd_smooth, popt, gaussian, aperiodic_simple, config["alpha_band"], floor_value=floor_value)

    if not peak_approved:
        return np.nan

    return paf

def philistine_iaf(psd, freq_bins, config):
    band = (freq_bins >= config["freq_range"][0]) & (freq_bins <= config["freq_range"][1])
    psd = psd[1:]
    freqs = freq_bins[1:]
    resolution = freqs[1] - freqs[0]

    sav_gol_polyorder = 5

    # As recommended in Philistine paper
    sav_gol_window_length = int(2.5 / resolution)
    if sav_gol_window_length % 2 == 0:
        sav_gol_window_length += 1

    if sav_gol_polyorder >= sav_gol_window_length:
        sav_gol_window_length = sav_gol_polyorder + 2

    fmin, fmax = config["alpha_band"][0], config["alpha_band"][1]
    
    psd_smooth = savgol_filter(psd, window_length=sav_gol_window_length, polyorder=sav_gol_polyorder)
    alpha_band = (freqs >= fmin) & (freqs <= fmax)

    eps = np.nextafter(0, 1)  # Smallest positive float
    _, _, r, _, _ = stats.linregress(np.log(freqs), np.log(np.maximum(psd_smooth, eps)))

    if r ** 2 > config["pink_ax_r2"]:
        return np.nan

    paf_idx = np.argmax(psd_smooth[alpha_band])
    paf = freqs[alpha_band][paf_idx]

    return paf


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_signal_debug(config, n_seconds=2):
    """Quick sanity-check plot for a single config. Call interactively."""
    signal, inst_freq = generate_signal(config)

    results, _,_ = run_window_analysis(signal[:int(n_seconds * config["fs"])], inst_freq[:int(n_seconds * config["fs"])], config)

    print("Estimates in first few windows:", {algo: est[:5] for algo, est in results.items()})

    fs = config["fs"]
    n_plot = int(n_seconds * fs)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(np.arange(n_plot) / fs, signal[:n_plot])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    f_ref = config["aperiodic_ref_freq"]
    beta  = config["aperiodic_exponent"]
    axes[0].set_title(
        f"carrier={config['carrier_freq']} Hz | "
        f"ref_freq={f_ref} Hz | "
        f"peak_snr={config['peak_snr_db']} dB (vs aperiodic @ {f_ref} Hz) | "
        f"β={beta} | aperiodic={'on' if config['has_aperiodic'] else 'off'}"
    )

    N = len(signal)

    if config["fft_method"] == "welch":
        freqs, psd = welch(signal, fs=fs, nperseg=2*fs)
    else:
        freqs = np.fft.rfftfreq(N, 1 / fs)
        psd = (np.abs(np.fft.rfft(signal)) ** 2) / (fs * N)
        psd[1:-1] *= 2  # Correct for one-sided PSD (except DC and Nyquist)
    mask = freqs <= 40
    axes[1].semilogy(freqs[mask], psd[mask], label="signal PSD")

    # Overlay expected aperiodic shape
    if config["has_aperiodic"]:
        ref = config["aperiodic_ref_power"]
        frot = config["f_rotation"]
        ap = ref * (freqs[mask] / frot) ** (-beta)
        axes[1].semilogy(freqs[mask], ap, "r--", label=f"aperiodic (β={beta})")

    # Mark aperiodic floor at the reference frequency
    ap_psd_at_ref = config["aperiodic_ref_power"] * (f_ref / config["f_rotation"]) ** (-beta)
    axes[1].axhline(ap_psd_at_ref, color="orange", linestyle="--",
                    label=f"aperiodic floor @ {f_ref} Hz")
    axes[1].axvline(f_ref, color="orange", alpha=0.5)

    # Mark peak target power level
    peak_power = ap_psd_at_ref * 10 ** (config["peak_snr_db"] / 10)
    axes[1].axhline(peak_power, color="g", linestyle=":",
                    label=f"peak target ({config['peak_snr_db']} dB vs ref)")
    axes[1].axvline(config["carrier_freq"], color="g", alpha=0.4,
                    label=f"carrier ({config['carrier_freq']} Hz)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    # config = {'fs': 10000.0, 'signal_length_sec': 20, 'freq_range': (1.0, 30.0), 'alpha_band': (5, 18), 'pink_ax_r2': 0.8, 'aperiodic_ref_power': 1.0, 
    #         'f_rotation': 1.0, 'aperiodic_ref_freq': 5.0, 'carrier_freq': 10.0, 'carrier_waveform': 'gaussian', 'mod_amp': 0.0, 'mod_freq': 0.0, 
    #         'aperiodic_exponent': 3, 'has_aperiodic': False, 'n_peaks': 1, 'peak_bw': 0.01, 'peak_snr_db': 20.0, 'noise_type': 'none', 'noise_snr_db': -10.0, 
    #         'window_length_sec': 5, "fft_method": 'fft'}

    # global RANDOM_SAMPLES
    # RANDOM_SAMPLES = np.random.randn(int(config["fs"] * config["signal_length_sec"]))
    
    # plot_signal_debug(config, n_seconds=6)