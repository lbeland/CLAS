"""Synthetic-signal generation for the algorithm comparison.

See :mod:`iaf_compare.config` for the meaning of the model parameters.

PSD/power quantities here are 10*log10(power) values: scaling is done as dB
addition/subtraction, dropping to a linear amplitude multiplier only where it
must apply to real FFT bin values (irfft).
"""
import numpy as np
from neurodsp.sim import sim_bursty_oscillation

from . import paths  # noqa: F401  -- side effect: puts SIMparam/code on sys.path

from fooof.sim.gen import gen_aperiodic, gen_periodic
from sims import gen_power_vals_fn

EPS = np.nextafter(0, 1)  # smallest positive float


def _power_to_db(power):
    """Linear power -> dB (10*log10)."""
    return 10 * np.log10(np.maximum(power, EPS))


def _db_to_amp(db):
    """Power ratio in dB -> linear amplitude multiplier (sqrt of the power ratio)."""
    return 10 ** (np.asarray(db) / 20)


def _constant_added_power(freqs_nz, aperiodic_params, peak_freq, peak_snr_db, peak_width):
    """Total extra power a "constant" mode peak adds on top of the aperiodic
    background: (aperiodic+peak total power) minus (aperiodic-only total
    power). Used to calibrate the burst peak to the same total power a
    constant-mode peak of this height/bw would add, so comparing
    stationarity isn't confounded by total energy."""
    ap_log = gen_aperiodic(freqs_nz, aperiodic_params)
    peak_log = gen_periodic(freqs_nz, [peak_freq, peak_snr_db / 10, peak_width])
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
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

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
    sig = sim_bursty_oscillation(n_samples / fs, fs, center_freq, burst_def="durations",
                                 burst_params={"n_cycles_burst": n_cycles_on, "n_cycles_off": n_cycles_off})

    current_power = np.mean(sig ** 2)
    scale = np.sqrt(target_power / max(current_power, EPS))
    return sig * scale


def generate_signal(config, rng):
    fs = config["fs"]
    n = int(config["signal_length_sec"] * fs)

    aperiodic_params = _aperiodic_params(
        config["aperiodic_ref_power_db"], config["f_rotation"], config["aperiodic_exponent"])
    freqs_nz = np.fft.rfftfreq(n, d=1 / fs)[1:]

    stationarity = config["stationarity"]
    n_peaks = config["n_peaks"]
    peak_freq = config["peak_freq"]
    has_peaks = n_peaks > 0
    gt_pf = peak_freq if has_peaks else np.nan

    # Aperiodic background (+ noise, + peaks if "constant"), as one coherent
    # FOOOF-model log-power spectrum: log10(power) = aperiodic + peak + noise.
    periodic_params = []
    if has_peaks and stationarity == "constant":
        # peak_snr_db is dB above the aperiodic floor AT peak_freq itself, so
        # it converts straight to FOOOF's height (log10-power units).
        periodic_params += [peak_freq, config["peak_snr_db"] / 10, config["peak_width"]]

        for idx in range(n_peaks - 1):
            extra_freq = peak_freq + (-1) ** idx * 2  # alternate sides, 2 Hz spacing
            # Extra peaks keep the old behaviour: 1/2 the main peak's power
            main_power_db = _aperiodic_floor_db(aperiodic_params, peak_freq) + config["peak_snr_db"]
            extra_power_db = main_power_db + _power_to_db(0.5)
            extra_height = (extra_power_db - _aperiodic_floor_db(aperiodic_params, extra_freq)) / 10
            periodic_params += [extra_freq, extra_height, config["peak_width"]]

    # fooof's gen_noise draws from numpy's unseeded *global* RNG, which would
    # make the noise realization different on every run (and, under a forked
    # Pool, correlated across workers) despite the per-trial seed. Route it
    # through the passed rng instead so the whole sweep is reproducible per
    # seed. Signature matches gen_noise(freqs, nlv).
    def _seeded_noise(freqs, nlv):
        return rng.normal(0.0, nlv, len(freqs))

    powers_nz = gen_power_vals_fn(
        freqs_nz,
        ap_kwargs={"aperiodic_params": aperiodic_params},
        pe_kwargs={"periodic_params": periodic_params},
        noise_kwargs={"nlv": config["noise_lv"]},
        ap_func=gen_aperiodic, pe_func=gen_periodic, noise_func=_seeded_noise,
    )
    signal = _spectrum_to_signal(n, fs, powers_nz, rng)

    # "bursty" carrier: genuine time-domain amplitude modulation, added on
    # top of the background since it has no FOOOF log-power equivalent.
    if has_peaks and stationarity == "bursty":
        main_power_db = _aperiodic_floor_db(aperiodic_params, peak_freq) + config["peak_snr_db"]
        # Same total power a "constant" mode peak of this height/bw would add
        target_power = _constant_added_power(
            freqs_nz, aperiodic_params, peak_freq, config["peak_snr_db"], config["peak_width"])
        signal += generate_burst_peak(n, fs, peak_freq, target_power)

        for idx in range(n_peaks - 1):
            extra_freq = peak_freq + (-1) ** idx * 2
            extra_power_db = main_power_db + _power_to_db(0.2)
            signal += generate_peak(n, fs, extra_freq, config["peak_width"], extra_power_db, rng)

    return signal, gt_pf
