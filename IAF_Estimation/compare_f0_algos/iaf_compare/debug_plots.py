"""Interactive sanity-check plot for a single simulation config."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from fooof.sim.gen import gen_aperiodic

from . import algorithms
from . import config as sweep_config
from .paths import OUTPUTS_DIR
from .plot_style import FIG_WIDTH, FIG_HEIGHT, FIGSIZE
from .signal_gen import generate_signal, _aperiodic_params, _aperiodic_floor_db
from .pipeline import run_window_analysis, compute_spectra

# pgf.texsystem defaults to xelatex, which isn't installed -- pdflatex is.
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def plot_signal_debug(config, n_seconds=2):
    """Quick sanity-check plot for a single config. Call interactively."""
    signal, gt_pf = generate_signal(config, np.random.default_rng(1))

    results, _, _ = run_window_analysis(signal[:int(n_seconds * config["fs"])], gt_pf, config)

    print("Estimates:", results)

    fs = config["fs"]
    n_plot = int(n_seconds * fs)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_WIDTH * 9 / 16))

    axes[0].plot(np.arange(n_plot) / fs, signal[:n_plot])
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)
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
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("PSD")
    axes[1].legend()
    axes[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

    axes[1].set_ylim(ap[-1], max(psd[mask]) * 1.5)

    plt.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUTS_DIR / f"debug_signal_{config['stationarity']}"
    plt.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(stem.with_suffix(".pgf"), bbox_inches="tight")


def default_alpha_fast_config():
    """The sweep's default OFAT condition (see iaf_compare.config), i.e. the
    same config every "Effect of X" plot holds everything else at."""
    config = {**sweep_config.FIXED, **sweep_config.DEFAULT}
    config["signal_length_sec"] = config["window_length_sec"]
    return config


def plot_alpha_fast_procedure(config=None, seed=1, save_name="alpha_fast_procedure"):
    """Thesis figure: the alpha_fast estimation procedure on one example
    window --
      (a) raw PSD
      (b) raw PSD in log-log scale, with both aperiodic fits overlaid (the
          initial fit, and the refined fit after rejecting samples above it)
      (c) whitened spectrum, its Savitzky-Golay-smoothed version, and the
          fitted Gaussian overlaid (center frequency named in its legend
          entry -- no axvline/axvspan)

    ``config`` defaults to the sweep's default OFAT condition (see
    :func:`default_alpha_fast_config`) -- pass a different one (e.g. a copy
    with a specific ``peak_snr_db``/``noise_lv``/... override) or a different
    ``seed`` to pick a cleaner example window, then call again until happy.
    """
    if config is None:
        config = default_alpha_fast_config()

    rng = np.random.default_rng(seed)
    signal, gt_pf = generate_signal(config, rng)

    fs = config["fs"]
    window_length = int(config["window_length_sec"] * fs)
    window = (signal if window_length >= len(signal)
              else signal[(len(signal) - window_length) // 2:(len(signal) + window_length) // 2])
    psd, _, _, freq_bins, _, _ = compute_spectra(window, window_length, fs, config)

    est_pf, diag = algorithms.alpha_fast(psd, freq_bins, config, return_diagnostics=True)

    freqs = diag["freqs"]

    fig, axes = plt.subplots(3,1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.5))

    # (a) Raw PSD
    ax = axes[0]
    ax.plot(freqs, diag["log_psd"], color="tab:blue", linewidth=1.2)
    ax.axvline(gt_pf, color="grey", linestyle="--", label=r"$f_0$"+f" = {gt_pf:.2f} Hz)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$\log_{10}$(Power)")
    ax.legend(loc="upper right")
    ax.set_title("(a)")

    # (b) log-log with both aperiodic fits
    ax = axes[1]
    ax.plot(diag["log_freqs"], diag["log_psd"], color="tab:blue", linewidth=1.0)
    ax.plot(diag["log_freqs"], diag["aperiodic_initial"], "--", color="tab:orange",
            linewidth=1.4, label="1th fit")
    ax.plot(diag["log_freqs"], diag["aperiodic_refined"], "-", color="tab:red",
            linewidth=1.4, label="2nd fit")
    ax.set_xlabel(r"$\log_{10}$(Frequency)")
    ax.set_ylabel(r"$\log_{10}$(Power)")
    ax.set_title("(b)")
    ax.legend(loc="upper right")

    # (c) Whitened spectrum + smoothed + Gaussian fit
    ax = axes[2]
    ax.plot(freqs, diag["psd_flat"], color="tab:blue", linewidth=0.8, alpha=0.5, label="whitened")
    ax.plot(freqs, diag["psd_smooth"], color="tab:blue", linewidth=1.4, label="smoothed")
    if diag["gaussian"] is not None:
        center = diag["popt"][1]
        ax.plot(freqs, diag["gaussian"], "--", color="tab:red", linewidth=1.4,
                label=f"Gaussian fit \n (centre = {center:.2f} Hz)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$\log_{10}$(Power)")
    ax.set_title("(c)")
    ax.legend(loc="upper right")

    for ax in axes:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

    fig.tight_layout()
    fig.align_labels()
    

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUTS_DIR / f"{save_name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUTS_DIR / f"{save_name}.pgf", bbox_inches="tight")
    return fig, axes
