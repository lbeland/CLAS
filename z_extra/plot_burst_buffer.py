"""
One-off test: plot the precomputed burst buffer (waveform + spectrum).

Reads the debug dump written once by StimControl::build_audio_buffers_()
at results/burst_buffer_debug.txt (first line: "# fs_audio=<Hz> num_octaves=<n>",
followed by one mono sample per line, normalized to peak amplitude 1).

Run from the repo root after the falcon graph has run StimControl at
least once:

    python extensions/processors/StimControl/plot_burst_buffer.py
"""
import sys
from pathlib import Path

from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "results" / "burst_buffer_debug.txt"

# 1/f ("pink") noise has PSD(f) ~ f^slope with slope == -1. Raw FFT magnitude
# is too noisy (single realization) to fit reliably, so we fit against a
# Welch PSD estimate instead.
TARGET_SLOPE = -1.0


def load_burst_buffer(path):
    with open(path) as f:
        header = f.readline()
    fs_audio = float(header.split("fs_audio=")[1].split()[0])
    samples = np.loadtxt(path, comments="#")
    return samples, fs_audio


def fit_slope(freqs, psd, f_min=1.0, f_max=None):
    """Least-squares fit of log10(psd) = slope * log10(f) + intercept."""
    if f_max is None:
        f_max = freqs[-1]
    mask = (freqs >= f_min) & (freqs <= f_max) & (psd > 0)
    slope, intercept = np.polyfit(np.log10(freqs[mask]), np.log10(psd[mask]), 1)
    return slope, intercept


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        raise SystemExit(f"No debug dump found at {path}. Run StimControl first.")

    samples, fs_audio = load_burst_buffer(path)
    n = len(samples)
    t = np.arange(n) / fs_audio

    print(f"Mean of time series: {np.mean(samples):.6f}, std: {np.std(samples):.6f}")

    freqs, psd = welch(samples, fs=fs_audio, nperseg=min(4096, n))
    slope, intercept = fit_slope(freqs, psd, f_min=1.0, f_max=fs_audio / 2)
    # fit_line = (10 ** intercept) * freqs ** slope

    print(f"Fitted PSD slope: {slope:.3f} (target for 1/f pink noise: {TARGET_SLOPE:.1f})")

    textwidth = 6.30045
    aspect_ratio = 9/16
    scale = 1.0
    width = textwidth * scale
    height = width * aspect_ratio
    fig, (ax_wave, ax_spec) = plt.subplots(1, 2, figsize=(width, height))

    ax_wave.plot(t[0:int(0.2*fs_audio)]*1000, samples[0:int(0.2*fs_audio)], linewidth=0.5)
    # ax_wave.plot(t*1000, samples, linewidth=0.5)

    ax_wave.set_xlabel("Time [ms]")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

    ax_spec.loglog(freqs[1:], psd[1:], linewidth=0.7)
    ax_spec.set_xlabel("Frequency [Hz]")
    ax_spec.set_ylabel("Power")
    ax_spec.grid(True, linestyle="--", linewidth=0.5, alpha=0.9)

    fig.tight_layout()
    plt.savefig("/home/linda/Documents/MA/plots/pink_noise.pgf", bbox_inches="tight")
    plt.savefig("/home/linda/Documents/MA/plots/pink_noise.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
