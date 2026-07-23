import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fooof import FOOOF
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, windows
from neurodsp.sim import sim_bursty_oscillation
from multitaper import MTSpec
from multitaper.utils import dpss
from pathlib import Path
from itertools import product
from tqdm import tqdm
import h5py
from scipy.signal import welch
from multiprocessing import Pool

BASE_FOLDER = Path(__file__).parent

sys.path.insert(0, str(BASE_FOLDER / "SIMparam" / "code"))
from fooof.sim.gen import gen_aperiodic, gen_periodic, gen_noise  # noqa: E402
from sims import gen_power_vals_fn  # noqa: E402

# ---------------------------------------------------------------------------
# Signal generation — FOOOF generative model
# ---------------------------------------------------------------------------
#
#   The "gaussian"-carrier spectrum is built exactly the way FOOOF assumes:
#       log10(power) = aperiodic(f) + peak(f) + noise(f)
#   via fooof.sim.gen (gen_aperiodic/gen_periodic/gen_noise, combined by
#   SIMparam/code/sims.gen_power_vals_fn), then synthesized into a real
#   signal by random-phase irfft. This removes any model mismatch against
#   the FOOOF/combine_* algorithms under test, and gives every condition
#   actual spectral noise (previously generated but never added).
#
#   "sine"/"burst" carriers are genuine time-domain waveforms (real spectral
#   leakage / amplitude modulation) with no equivalent in that log-power
#   model, so they're synthesized directly and added on top of a separately
#   generated aperiodic+noise background instead.
#
#   aperiodic_ref_power_db : float
#       PSD of the aperiodic component at f_rotation, in dB (10*log10(power)).
#       Acts as the global power anchor. 0 dB == 1 uV^2/Hz.
#
#   f_rotation : float
#       Spectral pivot frequency for the aperiodic shape (default: 1.0 Hz).
#       The aperiodic PSD equals aperiodic_ref_power_db exactly at this frequency.
#
#   peak_snr_db : float
#       Peak power relative to the aperiodic PSD AT THE PEAK'S OWN carrier_freq
#       (i.e. FOOOF's own peak-height convention -- no separate reference
#       frequency needed).
#         +10 dB -> peak clearly above its local background
#           0 dB -> peak power equals the local aperiodic floor
#          -6 dB -> peak buried below its local aperiodic -> stupid_max fails
#
#   noise_lv : float
#       Std dev of the per-bin noise added in log10-power space (gen_noise).
#
# ---------------------------------------------------------------------------

def main():
    fixed = {
        "fs":                   10000.0,
        "signal_length_sec":    30,
        "freq_range":           (0.01, 30.0),
        "alpha_band":           (5, 18),
        "pink_ax_r2":           0.9,
        # Aperiodic anchor power at f_rotation, in dB (0 dB == 1 uV^2/Hz)
        "aperiodic_ref_power_db": 0.0,
        # Spectral pivot — fix at 1 Hz so the shape is independent of carrier
        "f_rotation":           1.0,
        # Std dev of the per-bin log10-power noise (gen_noise)
        "noise_lv":             0.1,
    }

    default = {
        "carrier_freq":         10.32,
        # Peak shape in frequency domain
        "carrier_waveform":     "gaussian",   # "gaussian" | "sine" | "burst"
        # Aperiodic component
        "aperiodic_exponent":   2.0,          # β — slope of 1/f^β
        # Peak(s)
        "n_peaks":              1,
        "peak_bw":              0.5,          # Gaussian σ in Hz
        # KEY PARAM: peak power relative to the aperiodic floor at carrier_freq
        "peak_snr_db":          10.0,
        # Analysis
        "window_length_sec":    10,
    }

    cf = [value + 0.1*idx for idx, value in enumerate(np.arange(6, 16, 1.5))]

    sweeps = {
        "window_length_sec":    [5, 10, 20],
        "peak_snr_db":          [50, 20, 10, 5, 0],
        "peak_bw":              [0.1, 0.5, 1.0, 2.0, 4.0],
        "aperiodic_exponent":   [0, 1, 2, 3],
        "carrier_freq":         cf,
        "carrier_waveform":     ["gaussian", "sine", "burst"],
        "n_peaks":              [1, 2, 3],
    }

    # Sweeps for 0 peak default
    # sweeps = {
    #     "window_length_sec":    [5, 10, 20],
    #     "aperiodic_exponent":   [0, 1, 2, 3],
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

    # # Cross-product groups: each dict defines one joint sweep.
    # # All params not listed stay at their default value.
    # cross_sweeps = [
    #     {"peak_snr_db":        [20, 10, 3, 0, -3],
    #     "peak_bw":            [0.1, 0.5, 1.0, 2.0, 4.0]},

    #     {"carrier_freq":       cf,
    #     "aperiodic_exponent": [1, 2, 3]},
    # ]

    # for cross in cross_sweeps:
    #     params = list(cross.keys())
    #     for combo in product(*cross.values()):
    #         config = {**fixed, **default, **dict(zip(params, combo))}
    #         key = tuple(sorted(config.items(), key=lambda x: str(x)))
    #         if key not in seen:
    #             seen.add(key)
    #             conditions.append(config)
    

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
#
# dB helpers — all PSD/power quantities here are 10*log10(power) values.
# Scaling and normalisation math is done as dB addition/subtraction; we only
# drop to a linear amplitude multiplier at the point where it has to be
# applied to actual FFT bin values (irfft needs real sample values).

_EPS = np.nextafter(0, 1)

def _power_to_db(power):
    """Linear power -> dB (10*log10)."""
    return 10 * np.log10(np.maximum(power, _EPS))

def _db_to_amp(db):
    """Power ratio in dB -> linear amplitude multiplier (sqrt of the power ratio)."""
    return 10 ** (np.asarray(db) / 20)


def _aperiodic_params(ref_power_db, f_rotation, exponent):
    """Convert (PSD in dB at f_rotation, exponent) to fooof's gen_aperiodic
    'fixed'-mode params [offset, exponent], where offset = log10(power) at 1 Hz."""
    offset = ref_power_db / 10 + exponent * np.log10(f_rotation)
    return [offset, exponent]


def _aperiodic_floor_db(aperiodic_params, freq):
    """PSD (dB) of the aperiodic component at a single frequency."""
    return 10 * gen_aperiodic(np.asarray([freq]), aperiodic_params)[0]


def _spectrum_to_signal(n_samples, fs, powers_nz, rng):
    """One-sided target power spectrum (non-DC bins) -> real time signal via
    random-phase irfft. DC bin is left at zero (the aperiodic model is
    undefined at f=0).

    The /2 matches generate_peak/generate_sine_peak/generate_burst_peak's own
    `- _power_to_db(2)`: it pre-compensates for the one-sided PSD doubling
    (`psd[1:-1] *= 2`) applied by every downstream PSD estimate in this file,
    so `powers_nz` lands exactly on target after that correction.
    """
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
    amp = np.zeros_like(freqs)
    amp[1:] = np.sqrt(powers_nz * fs * n_samples / 2)
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    X = amp * np.exp(1j * phases)
    return np.fft.irfft(X, n=n_samples)


def generate_peak(n_samples, fs, center_freq, bw, peak_psd_db, rng):
    """
    Generate a narrowband oscillation with a Gaussian spectrum.
    Scaled so that PSD at center_freq equals peak_psd_db — consistent with
    how peak_snr_db is defined (i.e. PSD at the peak tip, not integrated power).

    Parameters
    ----------
    n_samples    : int
    fs           : float
    center_freq  : float   Hz
    bw           : float   Gaussian σ in Hz
    peak_psd_db  : float   desired PSD at center_freq, in dB (10*log10(power))

    Returns
    -------
    signal : np.ndarray  (n_samples,)
    """
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)

    # Gaussian amplitude envelope (unit peak at center_freq)
    amp_envelope = np.exp(-0.5 * ((freqs - center_freq) / bw) ** 2)

    # Current PSD at center bin = |amp_envelope[k]|² / (fs * n_samples)
    # At center: amp_envelope = 1.0, so current_psd_at_center = 1 / (fs * n_samples)
    # Scale so that PSD at center equals peak_psd_db (the /2 target accounts
    # for the one-sided PSD correction applied downstream).
    target_psd_db = peak_psd_db - _power_to_db(2)
    target_amp_at_center = _db_to_amp(target_psd_db + _power_to_db(fs * n_samples))
    amp_envelope *= target_amp_at_center   # peak of envelope hits target, shoulders scale with it

    phases = rng.uniform(0, 2 * np.pi, len(amp_envelope))
    X = amp_envelope * np.exp(1j * phases)

    sig = np.fft.irfft(X, n=n_samples)
    return sig


def generate_sine_peak(n_samples, fs, center_freq, peak_power_db, t):
    """
    Pure sine whose one-sided PSD at center_freq equals peak_power_db (dB).
    Amplitude derived analytically: one-sided PSD = A²·N/(2·fs)
    => A = sqrt(2·fs·peak_power_lin / N)
    """
    A = _db_to_amp(peak_power_db + _power_to_db(2 * fs / n_samples))
    return A * np.sin(2 * np.pi * center_freq * t)


def generate_burst_peak(n_samples, fs, center_freq, peak_power_db,
                        n_cycles_on=10, n_cycles_off=30):
    """Bursty oscillation with desired average power (peak_power_db, dB)."""

    sig = sim_bursty_oscillation(n_samples/fs,fs, center_freq, burst_def="durations",
                                burst_params={"n_cycles_burst": n_cycles_on, "n_cycles_off": n_cycles_off})

    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    idx = np.argmin(np.abs(freqs - center_freq))

    # Target |X[k]| such that one-sided PSD = |X[k]|²/(fs*N)*2 == 10**(peak_power_db/10)
    target_psd_db = peak_power_db - _power_to_db(2)
    target_amp = _db_to_amp(target_psd_db + _power_to_db(fs * n_samples))
    current_amp = np.abs(spec[idx])
    spec *= target_amp / max(current_amp, _EPS)
    return np.fft.irfft(spec, n=n_samples)


def generate_signal(config, rng):
    fs = config["fs"]
    n  = int(config["signal_length_sec"] * fs)
    t  = np.arange(n) / fs
    inst_freqs = np.full(n, np.nan)

    aperiodic_params = _aperiodic_params(
        config["aperiodic_ref_power_db"], config["f_rotation"], config["aperiodic_exponent"])
    freqs_nz = np.fft.rfftfreq(n, d=1 / fs)[1:]

    waveform = config["carrier_waveform"]
    n_peaks  = config["n_peaks"]
    cf       = config["carrier_freq"]
    has_peaks = n_peaks > 0
    if has_peaks:
        inst_freqs[:] = cf

    # ------------------------------------------------------------------
    # 1+2. Aperiodic background (+ noise, + peaks if "gaussian"), as one
    # coherent FOOOF-model log-power spectrum: log10(power) = aperiodic
    # + peak + noise. Folding the peak(s) in here (rather than adding a
    # separately-generated peak signal) is what makes this match FOOOF's
    # own generative assumptions exactly.
    # ------------------------------------------------------------------
    periodic_params = []
    if has_peaks and waveform == "gaussian":
        # peak_snr_db is dB above the aperiodic floor AT cf itself, so it
        # converts straight to FOOOF's height (log10-power units).
        periodic_params += [cf, config["peak_snr_db"] / 10, config["peak_bw"]]

        for idx in range(n_peaks - 1):
            extra_freq = cf + (-1) ** idx * 2  # alternate sides, 2 Hz spacing
            # Extra peaks keep the old behaviour: 1/5 the main peak's
            # absolute power (-7 dB), wherever that lands locally.
            main_power_db = _aperiodic_floor_db(aperiodic_params, cf) + config["peak_snr_db"]
            extra_power_db = main_power_db + _power_to_db(0.2)
            extra_height = (extra_power_db - _aperiodic_floor_db(aperiodic_params, extra_freq)) / 10
            periodic_params += [extra_freq, extra_height, config["peak_bw"]]

    powers_nz = gen_power_vals_fn(
        freqs_nz,
        ap_kwargs={"aperiodic_params": aperiodic_params},
        pe_kwargs={"periodic_params": periodic_params},
        noise_kwargs={"nlv": config["noise_lv"]},
        ap_func=gen_aperiodic, pe_func=gen_periodic, noise_func=gen_noise,
    )
    signal = _spectrum_to_signal(n, fs, powers_nz, rng)

    # ------------------------------------------------------------------
    # "sine"/"burst" carriers: real time-domain waveforms, added on top of
    # the background above since they have no FOOOF log-power equivalent.
    # ------------------------------------------------------------------
    if has_peaks and waveform != "gaussian":
        main_power_db = _aperiodic_floor_db(aperiodic_params, cf) + config["peak_snr_db"]

        if waveform == "sine":
            signal += generate_sine_peak(n, fs, cf, main_power_db, t)
        elif waveform == "burst":
            signal += generate_burst_peak(n, fs, cf, main_power_db)

        for idx in range(n_peaks - 1):
            extra_freq = cf + (-1) ** idx * 2
            extra_power_db = main_power_db + _power_to_db(0.2)
            signal += generate_peak(n, fs, extra_freq, config["peak_bw"], extra_power_db, rng)

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

    target_resolution_hz = 1.0   # narrowest peak width you need to resolve
    nw = max(config["window_length_sec"] * target_resolution_hz / 3, 2)
    kspec = int(2 * nw - 1)       # use the max well-concentrated tapers at this nw

    # vn, lamb = dpss(window_length, nw, kspec)   # compute Slepian tapers once, reused every window

    step = max(window_length // 5, int(config["fs"]))  # at most 1s step
    for idx in range(window_length, signal_length, step):
        window = signal[idx - window_length:idx]
        # window = window * windows.flattop(len(window))  # taper to reduce spectral leakage?

        nperseg = int(min(window_length, 7 * config["fs"]))
        freq_bins_welch, psd_welch = welch(window, fs=config["fs"], nperseg=nperseg, noverlap=nperseg//1.5)

        freq_bins = np.fft.rfftfreq(window_length, 1 / config["fs"])
        X = np.fft.rfft(window, n=window_length)
        # Periodogram PSD: |X|^2 / (fs * N) — matches Welch units (power per Hz)
        psd = (np.abs(X) ** 2) / (config["fs"] * window_length)
        psd[1:-1] *= 2  # Correct for dropping negative freqs in one-sided spectrum (except DC and Nyquist)

        # mt = MTSpec(window, nw=nw, kspec=kspec, dt=1/config["fs"], vn=vn, lamb=lamb)
        freq_mt, psd_mt = None, None# mt.rspec()

        if first_freq_spectrum is None:
            first_freq_spectrum = (freq_bins, psd)
            first_window = window

        ground_true_freqs.append(inst_freq[idx - 1])

        for algo, estimate in run_algorithms(psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt, config).items():
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

def run_algorithms(psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt, config):
    return {
        "stupid_max":   stupid_max(psd, freq_bins, config),
        "fooof":        fooof(psd_welch, freq_bins_welch, config),
        "philistine":   philistine_iaf(psd, freq_bins, config),
        # "philistine_simple": philistine_iaf_simple(psd, freq_bins, config),
        "combine_complex":      combine_algo(psd, freq_bins, config),
        "combine_simple":       combine_simple(psd, freq_bins, config),
        # "simple_mt": combine_simple_mt(psd_mt, freq_mt, config),
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

    if not peak_sig:
        return False
    else:
        return True
    
def combine_simple_mt(psd, freq_bins, config):
    band = (freq_bins >= config["freq_range"][0]) & (freq_bins <= config["freq_range"][1])
    psd_band = psd[band]
    freqs = freq_bins[band]

    resolution = freqs[1] - freqs[0]
    sav_gol_window_length = int(1.5 / resolution)
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
        # print(std_gauss)
        return np.nan
    popt = [amp_guess, center_guess, std_gauss]

    gaussian = gaussian_peak(freqs, *popt) + floor_value
    
    peak_approved = approve_peak(freqs, psd_safe, psd_flat, psd_smooth, popt, gaussian, aperiodic_simple, config["alpha_band"], floor_value=floor_value)

    if not peak_approved:
        return np.nan

    return paf

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
    ratio_threshold = 0.0 # 1.0
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
    paf = freqs[alpha_band][max_bin]

    if 0 < max_bin < (psd_alpha_band.size - 1):
        y1, y2, y3 = psd_alpha_band[max_bin - 1], psd_alpha_band[max_bin], psd_alpha_band[max_bin + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            paf += delta * resolution


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
    popt = [amp_guess, paf, std_gauss]

    gaussian = gaussian_peak(freqs, *popt) + floor_value
    
    peak_approved = approve_peak(freqs, log_psd, psd_flat, psd_smooth, popt, gaussian, aperiodic_simple, config["alpha_band"], floor_value=floor_value)

    if not peak_approved:
        return np.nan

    return paf

def _safe_log10(x):
    """log10 floored at the smallest positive float, since smoothed PSD (d0)
    can dip slightly negative from Savitzky-Golay overshoot."""
    return np.log10(np.maximum(x, np.nextafter(0, 1)))


def _min_pow_threshold(freqs, psd, mpow=1.0):
    """Port of restingIAF's polyfit/polyval(...,S) minPow calculation.

    Fits log10(psd) ~ freqs (linear, not log-log) and returns, per bin, the
    fitted value plus `mpow` prediction-error std devs -- the background
    noise threshold a candidate peak must clear.
    """
    y = _safe_log10(psd)
    X = np.column_stack([freqs, np.ones_like(freqs)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    dof = len(y) - X.shape[1]
    sigma_hat = np.sqrt(np.sum(resid ** 2) / dof)
    xtx_inv = np.linalg.inv(X.T @ X)
    leverage = np.einsum('ij,jk,ik->i', X, xtx_inv, X)
    delta = sigma_hat * np.sqrt(1 + leverage)
    return yhat + mpow * delta


def _estimate_peak_freq(d0, d1, freqs, w, min_pow, min_diff):
    """Peak-frequency-only port of restingIAF's peakBounds.m.

    Finds the dominant alpha peak (if any) via 1st-derivative zero-crossings
    within the alpha window `w`, checks it against the local noise floor
    (`min_pow`) and against competing peaks (`min_diff`). Returns NaN if no
    peak clears the noise floor or none is clearly dominant.

    (restingIAF also derives alpha-band bounds, inflection points and Q/Qf
    from here for its centre-of-gravity estimate; those aren't needed for a
    peak-frequency-only comparison, so they're dropped.)
    """
    n = len(freqs)
    lower_alpha = int(np.argmin(np.abs(freqs - w[0])))
    upper_alpha = int(np.argmin(np.abs(freqs - w[1])))

    # downward (peak-like) zero-crossings in d1, searched within the alpha window
    lo, hi = max(lower_alpha - 1, 0), min(upper_alpha + 1, n - 2)
    idx = np.arange(lo, hi + 1)
    idx = idx[np.sign(d1[idx]) > np.sign(d1[idx + 1])]
    if len(idx) == 0:
        return np.nan
    maxima = np.where(d0[idx] >= d0[idx + 1], idx, idx + 1)

    bins, powers = maxima, d0[maxima]
    top = np.argmax(powers)
    top_bin, top_power = bins[top], powers[top]

    if _safe_log10(top_power) <= min_pow[top_bin]:
        return np.nan  # doesn't clear the background noise floor

    if len(bins) > 1:
        runner_up_power = np.partition(powers, -2)[-2]
        if top_power * (1 - min_diff) <= runner_up_power:
            return np.nan  # not clearly dominant over a competing peak

    return freqs[top_bin]


def philistine_iaf(psd, freq_bins, config):
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

    mpow = config.get("mpow", 0.5)
    mdiff = config.get("mdiff", 0.10)
    min_pow = _min_pow_threshold(freqs, psd_band, mpow)

    return _estimate_peak_freq(d0, d1, freqs, config["alpha_band"], min_pow, mdiff)

def philistine_iaf_simple(psd, freq_bins, config):
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

    mpow = config.get("mpow", 1.0)
    mdiff = config.get("mdiff", 0.20)
    min_pow = _min_pow_threshold(freqs, psd_band, mpow)

    return _estimate_peak_freq(d0, d1, freqs, config["alpha_band"], min_pow, mdiff)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_signal_debug(config, n_seconds=2):
    """Quick sanity-check plot for a single config. Call interactively."""
    signal, inst_freq = generate_signal(config, np.random.default_rng(1))

    results, _,_ = run_window_analysis(signal[:int(n_seconds * config["fs"])], inst_freq[:int(n_seconds * config["fs"])], config)

    print("Estimates in first few windows:", {algo: est[:5] for algo, est in results.items()})

    fs = config["fs"]
    n_plot = int(n_seconds * fs)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(np.arange(n_plot) / fs, signal[:n_plot])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    beta = config["aperiodic_exponent"]
    axes[0].set_title(
        f"carrier={config['carrier_freq']} Hz | "
        f"peak_snr={config['peak_snr_db']} dB (vs aperiodic @ carrier_freq) | "
        f"aperiodic β={beta} | noise_lv={config['noise_lv']}"
    )

    N = len(signal)

    freqs = np.fft.rfftfreq(N, 1 / fs)
    psd = (np.abs(np.fft.rfft(signal)) ** 2) / (fs * N)
    psd[1:-1] *= 2  # Correct for one-sided PSD (except DC and Nyquist)
    mask = freqs <= 40
    axes[1].semilogy(freqs[mask], psd[mask], label="signal PSD")

    # Overlay expected aperiodic shape
    aperiodic_params = _aperiodic_params(
        config["aperiodic_ref_power_db"], config["f_rotation"], beta)
    ap = 10 ** gen_aperiodic(freqs[mask], aperiodic_params)
    axes[1].semilogy(freqs[mask], ap, "r--", label=f"aperiodic (β={beta})")

    # Mark peak target power level (dB above the aperiodic floor at carrier_freq)
    cf = config["carrier_freq"]
    peak_power_db = _aperiodic_floor_db(aperiodic_params, cf) + config["peak_snr_db"]
    axes[1].axhline(10 ** (peak_power_db / 10), color="g", linestyle=":",
                    label=f"peak target ({config['peak_snr_db']} dB vs local aperiodic)")
    axes[1].axvline(cf, color="g", alpha=0.4, label=f"carrier ({cf} Hz)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    # config = {'fs': 10000.0, 'signal_length_sec': 20, 'freq_range': (1.0, 30.0), 'alpha_band': (5, 18), 'pink_ax_r2': 0.8, 'aperiodic_ref_power_db': 0.0,
    #         'f_rotation': 1.0, 'noise_lv': 0.1, 'carrier_freq': 14.0, 'carrier_waveform': 'sine',
    #         'aperiodic_exponent': 3, 'n_peaks': 1, 'peak_bw': 2, 'peak_snr_db': 20.0,
    #         'window_length_sec': 10,}

    # plot_signal_debug(config, n_seconds=20)