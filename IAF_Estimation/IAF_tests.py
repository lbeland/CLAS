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
from modal import modal

BASE_FOLDER = Path(__file__).parent

# Params excluded from the LaTeX conditions table (not a meaningful sweep to report)
_LATEX_TABLE_EXCLUDE = ("aperiodic_ref_power_db", "f_rotation", "pink_ax_r2")


def _latex_escape(text):
    return str(text).replace("_", r"\_")


def _latex_format_value(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, tuple):
        return "--".join(_latex_format_value(v) for v in value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:g}"
    return _latex_escape(value)


# Fixed params still worth reporting alongside the sweeps
_LATEX_TABLE_FIXED_PARAMS = ("fs", "freq_range", "alpha_band")


name_dict = {
    "fs": "Sampling frequency [Hz]",
    "signal_length_sec": "Signal length [s]",
    "freq_range": "Frequency range [Hz]",
    "alpha_band": "Alpha band [Hz]",
    "peak_freq": "Peak frequency [Hz]",
    "stationarity": "Stationarity",
    "aperiodic_exponent": "Aperiodic exponent",
    "n_peaks": "Number of peaks",
    "peak_bw": "Peak bandwidth [Hz]",
    "peak_snr_db": "Peak SNR [dB]",
    "window_length_sec": "Window length [s]",
    "noise_lv": "Noise level ", # std dev of per-bin noise in log10-power space
}


def save_conditions_latex_table(default, sweeps, out_path, fixed=None,
                                 fixed_params=_LATEX_TABLE_FIXED_PARAMS,
                                 exclude=_LATEX_TABLE_EXCLUDE):
    """Write a LaTeX table: one row per swept parameter listing its tested
    values (default in bold), plus one row per fixed param."""

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Simulation parameters}",
        r"\label{tab:iaf_conditions}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        "Parameter & Tested values \\\\",
        r"\midrule",
    ]

    if fixed:
        for param in fixed_params:
            if param in fixed:
                param_name = name_dict.get(param, param)
                lines.append(rf"{_latex_escape(param_name)} & \textbf{{ {_latex_format_value(fixed[param])} }} \\")

    rows = [(param, values) for param, values in sweeps.items() if param not in exclude]
    for param, values in rows:
        default_val = default.get(param)
        value_texts = []
        for val in values:
            text = _latex_format_value(val)
            if default_val is not None and val == default_val:
                text = rf"\textbf{{{text}}}"
            value_texts.append(text)
        lines.append(f"{_latex_escape(name_dict.get(param, param))} & {', '.join(value_texts)} \\\\")


    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_path.write_text("\n".join(lines) + "\n")

sys.path.insert(0, str(BASE_FOLDER / "SIMparam" / "code"))
from fooof.sim.gen import gen_aperiodic, gen_periodic, gen_noise  # noqa: E402
from sims import gen_power_vals_fn  # noqa: E402

# Signal generation — FOOOF generative model.
# "constant" stationarity: log10(power) = aperiodic(f) + peak(f) + noise(f),
# via fooof.sim.gen, synthesized into a real signal by random-phase irfft.
# "burst" stationarity: genuine time-domain amplitude modulation, no
# log-power equivalent, so it's synthesized directly and added on top of a
# separately generated aperiodic+noise background instead.
#
#   aperiodic_ref_power_db : PSD of the aperiodic component at f_rotation,
#       in dB (10*log10(power)). Global power anchor. 0 dB == 1 uV^2/Hz.
#   f_rotation : spectral pivot frequency for the aperiodic shape; the
#       aperiodic PSD equals aperiodic_ref_power_db exactly here.
#   peak_snr_db : peak power relative to the aperiodic PSD at the peak's own
#       peak_freq (FOOOF's own peak-height convention).
#   noise_lv : std dev of the per-bin noise added in log10-power space.

def main():
    fixed = {
        "fs":                   10000.0,
        "signal_length_sec":    30,
        "freq_range":           (0.01, 30.0),
        "alpha_band":           (5, 18),
        "pink_ax_r2":           0.9,
        "aperiodic_ref_power_db": 0.0,  # anchor power at f_rotation, dB (0 dB == 1 uV^2/Hz)
        "f_rotation":           1.0,    # pivot freq so aperiodic shape is independent of carrier
    }

    default = {
        "peak_freq":            np.float64(10.32),
        "stationarity":     "constant",   # "constant" | "burst"
        "aperiodic_exponent":   2.0,      # β — slope of 1/f^β
        "n_peaks":              1,
        "peak_bw":              0.5,      # Gaussian σ in Hz
        "peak_snr_db":          10.0,     # peak power relative to aperiodic floor at peak_freq
        "window_length_sec":    10,
        "noise_lv":             0.5,      # std dev of per-bin log10-power noise (gen_noise)
    }

    # window_length_sec is evaluated separately below (crossed against every
    # condition here, see "Window-length comparison"), not as an OFAT sweep entry.
    sweeps = {
        "peak_snr_db":          [0, 5, 10, 20, 50],
        "peak_bw":              [0.1, 0.5, 1.0, 2.0, 4.0],
        "aperiodic_exponent":   [0, 1, 2, 3],
        "peak_freq":            sorted([6,8,12,14] + [default["peak_freq"]]),
        "stationarity":         ["constant", "burst"],
        "n_peaks":              [1, 2, 3],
        "noise_lv":             [0, 0.2, 0.5, 1.0],
    }

    window_lengths_sec = [5, 10, 20]

    save_conditions_latex_table(
        default, {**sweeps, "window_length_sec": window_lengths_sec},
        BASE_FOLDER / "conditions_table.tex", fixed=fixed)

    seen = set()
    conditions = []

    for param, values in sweeps.items():
        sweep_default = {**default}

        for val in values:
            config = {**fixed, **sweep_default, param: val}
            # Generate each signal at exactly window_length_sec -- one
            # signal == one window == one trial, no sliding.
            config["signal_length_sec"] = config["window_length_sec"]
            key = tuple(sorted(config.items(), key=lambda x: str(x)))
            if key not in seen:
                seen.add(key)
                conditions.append(config)

    # # Cross-product groups: each dict defines one joint sweep.
    # # All params not listed stay at their default value.
    # cross_sweeps = [
    #     {"peak_snr_db":        [20, 10, 3, 0, -3],
    #     "peak_bw":            [0.1, 0.5, 1.0, 2.0, 4.0]},

    #     {"peak_freq":          peak_freq,
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

    N_SEEDS = 20  # start small; increase once runtime is acceptable

    conditions_with_seeds = [
        (i * N_SEEDS + seed, cfg, seed)
        for i, cfg in enumerate(conditions)
        for seed in range(N_SEEDS)
    ]
    n_base_trials = len(conditions_with_seeds)

    # Window-length comparison: re-evaluate every condition above at several
    # window lengths, sharing ONE parent signal per (condition, seed)
    next_idx = n_base_trials
    window_length_groups = []
    for cfg in conditions:
        for seed in range(N_SEEDS):
            cond_indices = list(range(next_idx, next_idx + len(window_lengths_sec)))
            next_idx += len(window_lengths_sec)
            window_length_groups.append((cond_indices, cfg, seed, window_lengths_sec))

    print(f"Total window-length trials: {len(window_length_groups) * len(window_lengths_sec)}")

    n_workers = 5 # max(1, cpu_count() - 1)
    with Pool(n_workers) as pool:
        base_results = list(tqdm(
            pool.imap(process_condition, conditions_with_seeds),
            total=n_base_trials,
            desc="Processing base conditions",
        ))
        window_length_result_groups = list(tqdm(
            pool.imap(process_window_length_condition, window_length_groups),
            total=len(window_length_groups),
            desc="Processing window-length conditions",
        ))

    all_results = base_results + [row for group in window_length_result_groups for row in group]

    # Pool every seed sharing the same condition (identical stored_config)
    # into one entry: a single example spectrum/window (all seeds of a
    # condition look alike, so N_SEEDS copies would be redundant) plus,
    # per algo, the full list of per-seed estimates.
    pooled_samples = {}
    example_by_condition = {}
    gt_by_condition = {}
    for _, config, gt, estimates_per_algo, spectrum_info in all_results:
        condition_key = tuple(sorted(config.items(), key=lambda x: str(x)))
        algo_samples = pooled_samples.setdefault(condition_key, {})
        for algo, est_pf in estimates_per_algo.items():
            samples = algo_samples.setdefault(algo, {"est": [], "gt": []})
            samples["est"].append(float(est_pf))
            samples["gt"].append(gt)
        example_by_condition.setdefault(condition_key, spectrum_info)
        gt_by_condition.setdefault(condition_key, gt)

    with h5py.File(BASE_FOLDER / "iaf_results.h5", "w") as hf:
        hf.attrs["created"] = str(pd.Timestamp.now())
        conds_grp = hf.create_group("conditions")

        for cond_idx, (condition_key, algo_samples) in enumerate(tqdm(
                pooled_samples.items(), total=len(pooled_samples), desc="Writing HDF5")):
            grp = conds_grp.create_group(f"{cond_idx:04d}")

            for k, v in dict(condition_key).items():
                if isinstance(v, tuple):
                    grp.attrs[k] = list(v)
                else:
                    grp.attrs[k] = v if v is not None else "none"

            (freq_bins, psd), first_window = example_by_condition[condition_key]
            grp.create_dataset("spectrum_psd",       data=psd,          compression="gzip")
            grp.create_dataset("spectrum_freq_bins", data=freq_bins,    compression="gzip")
            grp.create_dataset("first_window",       data=first_window, compression="gzip")
            grp.create_dataset("ground_truth",       data=np.array([gt_by_condition[condition_key]]), compression="gzip")

            for algo, samples in algo_samples.items():
                metrics = compute_pooled_metrics(samples["est"], samples["gt"])
                algo_grp = grp.create_group(algo)
                algo_grp.create_dataset("estimates", data=metrics.pop("estimates"), compression="gzip")
                algo_grp.create_dataset("errors",    data=metrics.pop("errors"),    compression="gzip")
                for metric, val in metrics.items():
                    algo_grp.attrs[metric] = val


# Signal generation
#
# dB helpers — PSD/power quantities here are 10*log10(power) values. Scaling
# is done as dB addition/subtraction, dropping to a linear amplitude
# multiplier only where it must apply to real FFT bin values (irfft).

_EPS = np.nextafter(0, 1)

def _power_to_db(power):
    """Linear power -> dB (10*log10)."""
    return 10 * np.log10(np.maximum(power, _EPS))

def _db_to_amp(db):
    """Power ratio in dB -> linear amplitude multiplier (sqrt of the power ratio)."""
    return 10 ** (np.asarray(db) / 20)


def _constant_added_power(freqs_nz, aperiodic_params, peak_freq, peak_snr_db, peak_bw):
    """Total extra power a "constant" mode peak adds on top of the aperiodic
    background: (aperiodic+peak total power) minus (aperiodic-only total
    power). Used to calibrate the burst peak to the same total power a
    constant-mode peak of this height/bw would add, so comparing
    stationarity isn't confounded by total energy."""
    ap_log = gen_aperiodic(freqs_nz, aperiodic_params)
    peak_log = gen_periodic(freqs_nz, [peak_freq, peak_snr_db / 10, peak_bw])
    df = freqs_nz[1] - freqs_nz[0]
    power_with_peak = np.sum(10 ** (ap_log + peak_log)) * df
    power_ap_only = np.sum(10 ** ap_log) * df
    return power_with_peak - power_ap_only


def _aperiodic_params(ref_power_db, f_rotation, exponent):
    """Convert (PSD in dB at f_rotation, exponent) to fooof's gen_aperiodic
    'fixed'-mode params [offset, exponent]."""
    offset = ref_power_db / 10 + exponent * np.log10(f_rotation)
    return [offset, exponent]


def _aperiodic_floor_db(aperiodic_params, freq):
    """PSD (dB) of the aperiodic component at a single frequency."""
    return 10 * gen_aperiodic(np.asarray([freq]), aperiodic_params)[0]


def _spectrum_to_signal(n_samples, fs, powers_nz, rng):
    """One-sided target power spectrum (non-DC bins) -> real time signal via
    random-phase irfft. DC bin left at zero (aperiodic model undefined at
    f=0). The /2 pre-compensates for the one-sided PSD doubling
    (`psd[1:-1] *= 2`) applied by every downstream PSD estimate, matching
    generate_peak's `- _power_to_db(2)`."""
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
    amp = np.zeros_like(freqs)
    amp[1:] = np.sqrt(powers_nz * fs * n_samples / 2)
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    X = amp * np.exp(1j * phases)
    return np.fft.irfft(X, n=n_samples)


def generate_peak(n_samples, fs, center_freq, bw, peak_psd_db, rng):
    """Narrowband oscillation with a Gaussian spectrum, scaled so PSD at
    center_freq equals peak_psd_db (dB, 10*log10(power))."""
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)

    # Gaussian amplitude envelope (unit peak at center_freq)
    amp_envelope = np.exp(-0.5 * ((freqs - center_freq) / bw) ** 2)

    # Scale so PSD at center equals peak_psd_db (/2 accounts for the
    # one-sided PSD correction applied downstream)
    target_psd_db = peak_psd_db - _power_to_db(2)
    target_amp_at_center = _db_to_amp(target_psd_db + _power_to_db(fs * n_samples))
    amp_envelope *= target_amp_at_center

    phases = rng.uniform(0, 2 * np.pi, len(amp_envelope))
    X = amp_envelope * np.exp(1j * phases)

    sig = np.fft.irfft(X, n=n_samples)
    return sig


def generate_burst_peak(n_samples, fs, center_freq, target_power,
                        n_cycles_on=10, n_cycles_off=30):
    """Bursty oscillation with average power matched to target_power
    (linear), via time-domain RMS rather than a single FFT bin, since a
    burst's energy is smeared across many sidebands rather than
    concentrated at center_freq."""
    sig = sim_bursty_oscillation(n_samples/fs, fs, center_freq, burst_def="durations",
                                burst_params={"n_cycles_burst": n_cycles_on, "n_cycles_off": n_cycles_off})

    current_power = np.mean(sig ** 2)
    scale = np.sqrt(target_power / max(current_power, _EPS))
    return sig * scale
# def generate_burst_peak(n_samples, fs, center_freq, peak_power_db,
#                         n_cycles_on=10, n_cycles_off=30):
#     """Bursty oscillation with desired average power (peak_power_db, dB)."""

#     sig = sim_bursty_oscillation(n_samples/fs,fs, center_freq, burst_def="durations",
#                                 burst_params={"n_cycles_burst": n_cycles_on, "n_cycles_off": n_cycles_off})

#     spec = np.fft.rfft(sig)
#     freqs = np.fft.rfftfreq(n_samples, d=1/fs)
#     idx = np.argmin(np.abs(freqs - center_freq))

#     # Target |X[k]| such that one-sided PSD = |X[k]|²/(fs*N)*2 == 10**(peak_power_db/10)
#     target_psd_db = peak_power_db - _power_to_db(2)
#     target_amp = _db_to_amp(target_psd_db + _power_to_db(fs * n_samples))
#     current_amp = np.abs(spec[idx])
#     spec *= target_amp / max(current_amp, _EPS)
#     return np.fft.irfft(spec, n=n_samples)


def generate_signal(config, rng):
    fs = config["fs"]
    n  = int(config["signal_length_sec"] * fs)

    aperiodic_params = _aperiodic_params(
        config["aperiodic_ref_power_db"], config["f_rotation"], config["aperiodic_exponent"])
    freqs_nz = np.fft.rfftfreq(n, d=1 / fs)[1:]

    stationarity = config["stationarity"]
    n_peaks  = config["n_peaks"]
    peak_freq = config["peak_freq"]
    has_peaks = n_peaks > 0
    gt_pf = peak_freq if has_peaks else np.nan

    # Aperiodic background (+ noise, + peaks if "constant"), as one coherent
    # FOOOF-model log-power spectrum: log10(power) = aperiodic + peak + noise.
    periodic_params = []
    if has_peaks and stationarity == "constant":
        # peak_snr_db is dB above the aperiodic floor AT peak_freq itself, so
        # it converts straight to FOOOF's height (log10-power units).
        periodic_params += [peak_freq, config["peak_snr_db"] / 10, config["peak_bw"]]

        for idx in range(n_peaks - 1):
            extra_freq = peak_freq + (-1) ** idx * 2  # alternate sides, 2 Hz spacing
            # Extra peaks keep the old behaviour: 1/2 the main peak's power
            main_power_db = _aperiodic_floor_db(aperiodic_params, peak_freq) + config["peak_snr_db"]
            extra_power_db = main_power_db + _power_to_db(0.5)
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

    # "burst" carrier: genuine time-domain amplitude modulation, added on
    # top of the background since it has no FOOOF log-power equivalent.
    if has_peaks and stationarity == "burst":
        main_power_db = _aperiodic_floor_db(aperiodic_params, peak_freq) + config["peak_snr_db"]
        # Same total power a "constant" mode peak of this height/bw would add
        target_power = _constant_added_power(
            freqs_nz, aperiodic_params, peak_freq, config["peak_snr_db"], config["peak_bw"])
        signal += generate_burst_peak(n, fs, peak_freq, target_power)

        for idx in range(n_peaks - 1):
            extra_freq = peak_freq + (-1) ** idx * 2
            extra_power_db = main_power_db + _power_to_db(0.2)
            signal += generate_peak(n, fs, extra_freq, config["peak_bw"], extra_power_db, rng)

    return signal, gt_pf


# Pipeline

def process_condition(args):
    cond_idx, config, seed = args
    try:
        rng = np.random.default_rng(seed)
        signal, gt_pf = generate_signal(config, rng)
        estimates_per_algo, gt_pf, spectrum_info = run_window_analysis(signal, gt_pf, config)
        stored_config = {**config, "trial_source": "base"}
        return cond_idx, stored_config, gt_pf, estimates_per_algo, spectrum_info
    except Exception as e:
        print(f"Error processing condition\n{config}\n: {e}")
        raise


def process_window_length_condition(args):
    """Like process_condition, but shares ONE parent signal -- generated at
    the longest window_length_sec being compared -- across every variant of
    this (condition, seed), each analyzing a nested prefix sub-window. This
    isolates the effect of window length itself, instead of confounding it
    with a fresh random draw."""
    cond_indices, base_config, seed, window_lengths_sec = args
    try:
        rng = np.random.default_rng(seed)
        parent_config = {**base_config, "signal_length_sec": max(window_lengths_sec)}
        signal, gt_pf = generate_signal(parent_config, rng)

        results = []
        for cond_idx, window_length_sec in zip(cond_indices, window_lengths_sec):
            config = {**parent_config, "window_length_sec": window_length_sec}
            estimates_per_algo, gt_pf_i, spectrum_info = run_window_analysis(signal, gt_pf, config)
            stored_config = {**config, "trial_source": "window_length_sweep"}
            results.append((cond_idx, stored_config, gt_pf_i, estimates_per_algo, spectrum_info))
        return results
    except Exception as e:
        print(f"Error processing window-length condition group\n{base_config}\n: {e}")
        raise


def run_window_analysis(signal, gt_pf, config):
    """Analyze the signal as a single window covering its full length.
    (Signals are generated at exactly window_length_sec, so there's nothing
    to slide through -- one signal == one window == one trial.)"""
    window_length = int(config["window_length_sec"] * config["fs"])
    if window_length > len(signal):
        window = signal[(len(signal)-window_length)//2:(len(signal)+window_length)//2]
    else:
        window = signal
    # window = window * windows.flattop(len(window))  # taper to reduce spectral leakage?

    target_resolution_hz = 1.0   # narrowest peak width you need to resolve
    nw = max(config["window_length_sec"] * target_resolution_hz / 3, 2)
    kspec = int(2 * nw - 1)       # use the max well-concentrated tapers at this nw

    # vn, lamb = dpss(window_length, nw, kspec)   # compute Slepian tapers once, reused every window

    nperseg = int(min(window_length, 7 * config["fs"]))
    freq_bins_welch, psd_welch = welch(window, fs=config["fs"], nperseg=nperseg, noverlap=nperseg//1.5)

    freq_bins = np.fft.rfftfreq(window_length, 1 / config["fs"])
    X = np.fft.rfft(window, n=window_length)
    # Periodogram PSD: |X|^2 / (fs * N) — matches Welch units (power per Hz)
    psd = (np.abs(X) ** 2) / (config["fs"] * window_length)
    psd[1:-1] *= 2  # Correct for dropping negative freqs in one-sided spectrum (except DC and Nyquist)

    # mt = MTSpec(window, nw=nw, kspec=kspec, dt=1/config["fs"], vn=vn, lamb=lamb)
    freq_mt, psd_mt = None, None# mt.rspec()

    estimates_per_algo = run_algorithms(
        window, psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt, config)

    return estimates_per_algo, gt_pf, ((freq_bins, psd), window)


def compute_pooled_metrics(estimates, ground_truths):
    """Confusion-matrix + error metrics pooled over many (estimate, gt)
    pairs -- e.g. every seed of one condition -- since rmse and f1_score
    are nonlinear and can't be recovered by averaging n=1 per-trial values
    afterwards. errors stays nan-padded to len(estimates) (not compacted)
    so it stays index-aligned with estimates for per-seed storage."""
    estimates = np.asarray(estimates, dtype=float)
    ground_truths = np.asarray(ground_truths, dtype=float)
    errors = np.full(estimates.shape, np.nan)
    fp = fn = tn = tp = 0

    for i, (est, gt) in enumerate(zip(estimates, ground_truths)):
        est_none, gt_none = np.isnan(est), np.isnan(gt)
        if gt_none and est_none:
            tn += 1
        elif gt_none and not est_none:
            fp += 1
            errors[i] = est
        elif not gt_none and est_none:
            fn += 1
        else:
            tp += 1
            errors[i] = est - gt

    abs_errors = np.abs(errors[~np.isnan(errors)])
    n = len(estimates)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    accuracy = (tp + tn) / n if n > 0 else np.nan
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

    return {
        "estimates":      estimates,
        "errors":         errors,
        "mae":            np.mean(abs_errors) if abs_errors.size else np.nan,
        "rmse":           np.sqrt(np.mean(abs_errors ** 2)) if abs_errors.size else np.nan,
        "std":            np.std(abs_errors) if abs_errors.size else np.nan,
        "fn":             fn,
        "fp":             fp,
        "n":              n,
        "fail_rate":      (fn+fp) / n,
        "accuracy":       accuracy,
        "precision":      precision,
        "recall":         recall,
        "f1_score":       f1_score,
    }


# Algorithms

def run_algorithms(window, psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt, config):
    return {
        "stupid_max":   stupid_max(psd, freq_bins, config),
        "fooof":        fooof(psd_welch, freq_bins_welch, config),
        "philistine":   philistine_iaf(psd, freq_bins, config),
        # "modal":        modal_iaf(window, config),
        "combine_complex":      combine_algo(psd, freq_bins, config),
        "combine_simple":       combine_simple(psd, freq_bins, config),
        # "simple_mt": combine_simple_mt(psd_mt, freq_mt, config),
    }


def stupid_max(psd, freq_bins, config):
    band = (freq_bins >= config["alpha_band"][0]) & (freq_bins <= config["alpha_band"][1])
    psd_band = psd[band]
    max_bin = np.argmax(psd_band)
    freq_bins_band = freq_bins[band]
    est_pf = freq_bins_band[max_bin]
    # at the band's lower edge: reject if not above the bin just before it
    # (avoids picking low-freq noise when no clear peak is present)
    if est_pf == freq_bins_band[0]:
        if psd_band[max_bin] < psd[freq_bins < config["alpha_band"][0]][-1]:
            est_pf = np.nan
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

# def modal_iaf(window, config):
#     try:
#         # 6-cycle wavelets need >=6 periods to fit in the window; frequencies
#         # below that (e.g. FOOOF's 0.01 Hz aperiodic-fit floor) blow up the
#         # wavelet length far past the window length.
#         min_wavefreq = config["alpha_band"][0]
#         params = {
#             "srate": config["fs"],
#             "wavefreqs": np.arange(min_wavefreq, config["freq_range"][1], 1.0),
#             "local_winsize_sec": [min(10, len(window)/config["fs"])],
#             "wavecycles": 6,
#             "crop_fs": True,
#         }
#         iaf = modal(window, params)
#         if all(np.isnan(iaf)):
#             return np.nan
#         return np.nanmean(iaf)
#     except Exception as e:
#         print(f"Modal fitting error: {e}")
#         return np.nan

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

    fooof_model = FOOOF(peak_width_limits=(0.1, 4.0), max_n_peaks=3, min_peak_height=0.0,
               peak_threshold=2.0, aperiodic_mode="fixed", verbose=False)
    try:
        fooof_model.fit(freqs, psd_band, config["freq_range"])
    except Exception as e:
        print(f"FOOOF fitting error: {e}")
        return np.nan

    offset, exponent = fooof_model.aperiodic_params_
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
    est_pf = freqs[alpha_band][max_bin]
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
            return est_pf + delta * resolution
        
    return est_pf


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
    est_pf = freqs[alpha_band][max_bin]

    if 0 < max_bin < (psd_alpha_band.size - 1):
        y1, y2, y3 = psd_alpha_band[max_bin - 1], psd_alpha_band[max_bin], psd_alpha_band[max_bin + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            est_pf += delta * resolution


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

    return est_pf

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
    
    peak_approved = approve_peak(freqs, log_psd, psd_flat, psd_smooth, popt, gaussian, aperiodic_simple, config["alpha_band"], floor_value=floor_value)

    if not peak_approved:
        return np.nan

    return est_pf

def _safe_log10(x):
    """log10 floored at the smallest positive float (smoothed PSD can dip
    slightly negative from Savitzky-Golay overshoot)."""
    return np.log10(np.maximum(x, np.nextafter(0, 1)))


def _min_pow_threshold(freqs, psd, mpow=1.0):
    """Port of restingIAF's polyfit/polyval(...,S) minPow calculation. Fits
    log10(psd) ~ freqs (linear) and returns, per bin, the fitted value plus
    `mpow` prediction-error std devs -- the noise threshold a candidate
    peak must clear."""
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
    """Peak-frequency-only port of restingIAF's peakBounds.m. Finds the
    dominant alpha peak (if any) via 1st-derivative zero-crossings within
    window `w`, checked against the noise floor (`min_pow`) and competing
    peaks (`min_diff`). Returns NaN if no peak clears the floor or none is
    clearly dominant."""
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

    # if len(bins) > 1:
    #     runner_up_power = np.partition(powers, -2)[-2]
    #     if top_power * (1 - min_diff) <= runner_up_power:
    #         return np.nan  # not clearly dominant over a competing peak

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

    mpow = config.get("mpow", 0.4)
    mdiff = config.get("mdiff", 0.10)
    min_pow = _min_pow_threshold(freqs, psd_band, mpow)

    return _estimate_peak_freq(d0, d1, freqs, config["alpha_band"], min_pow, mdiff)

# Plotting

def plot_signal_debug(config, n_seconds=2):
    """Quick sanity-check plot for a single config. Call interactively."""
    signal, gt_pf = generate_signal(config, np.random.default_rng(1))

    results, _,_ = run_window_analysis(signal[:int(n_seconds * config["fs"])], gt_pf, config)

    print("Estimates:", results)

    fs = config["fs"]
    n_plot = int(n_seconds * fs)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(np.arange(n_plot) / fs, signal[:n_plot])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    beta = config["aperiodic_exponent"]
    axes[0].set_title(
        f"peak_freq={config['peak_freq']} Hz | "
        f"peak_snr={config['peak_snr_db']} dB (vs aperiodic @ peak_freq) | "
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

    # Mark peak target power level (dB above the aperiodic floor at peak_freq)
    peak_freq = config["peak_freq"]
    peak_power_db = _aperiodic_floor_db(aperiodic_params, peak_freq) + config["peak_snr_db"]
    axes[1].axhline(10 ** (peak_power_db / 10), color="g", linestyle=":",
                    label=f"peak target ({config['peak_snr_db']} dB vs local aperiodic)")
    axes[1].axvline(peak_freq, color="g", alpha=0.4, label=f"peak ({peak_freq} Hz)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].legend(fontsize=8)

    axes[1].set_ylim(ap[-1], max(psd[mask]) * 1.5)

    plt.tight_layout()
    plt.savefig(f"debug_signal_{config['stationarity']}.png", dpi=300)



if __name__ == "__main__":
    main()

    # config = {'fs': 10000.0, 'signal_length_sec': 20, 'freq_range': (1.0, 30.0), 'alpha_band': (5, 18), 'pink_ax_r2': 0.8, 'aperiodic_ref_power_db': 0.0,
    #         'f_rotation': 1.0, 'noise_lv': 0.1, 'peak_freq': 14.0, 'stationarity': 'burst',
    #         'aperiodic_exponent': 2, 'n_peaks': 1, 'peak_bw': 0.5, 'peak_snr_db': 20.0,
    #         'window_length_sec': 10,}

    # plot_signal_debug(config, n_seconds=20)
    # plot_signal_debug({**config, 'stationarity': 'constant'}, n_seconds=20)

    # plt.show()