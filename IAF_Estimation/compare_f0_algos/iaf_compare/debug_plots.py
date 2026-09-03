"""Interactive sanity-check plot for a single simulation config."""
import numpy as np
import matplotlib.pyplot as plt
from fooof.sim.gen import gen_aperiodic

from .paths import OUTPUTS_DIR
from .signal_gen import generate_signal, _aperiodic_params, _aperiodic_floor_db
from .pipeline import run_window_analysis


def plot_signal_debug(config, n_seconds=2):
    """Quick sanity-check plot for a single config. Call interactively."""
    signal, gt_pf = generate_signal(config, np.random.default_rng(1))

    results, _, _ = run_window_analysis(signal[:int(n_seconds * config["fs"])], gt_pf, config)

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
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUTS_DIR / f"debug_signal_{config['stationarity']}.png", dpi=300)
