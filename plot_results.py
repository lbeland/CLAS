import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meegkit.phase import ECHT
from scipy.signal import hilbert
from read_output import get_signal_data

def wrap_phase_deg(x_deg):
    return (x_deg + 180.0) % 360.0 - 180.0

def plot_results(fs, f0, orig_node):
    # Load data
    if orig_node == "Producer":
        samples_orig = get_signal_data('rt_c_results/Serializer1.0_Producer.out.0.bin')
    elif orig_node == "SourceClient":
        samples_orig = get_signal_data('rt_c_results/Serializer1.0_SourceClient.out.0.bin')

    n = len(samples_orig)

    if samples_orig is None:
        print("No original signal data found. Exiting.")
        return
    
    if os.path.exists("simulated_signal.npy"):
        loaded = np.load("simulated_signal.npy")
        assert samples_orig[0,0] == loaded["value"][0], "Loaded simulated signal does not match original signal from file"
        true_phase = loaded["phase"]
        true_inst_freq = loaded["inst_freq"]
    else:
        true_phase = None
        true_inst_freq = None

    samples_filtered = get_signal_data('rt_c_results/Serializer2.0_BandpassFilter.out.0.bin')
    samples_phase = get_signal_data('rt_c_results/Serializer3.0_PhaseEstimator.out.0.bin')
    samples_real = get_signal_data('rt_c_results/Serializer3.1_PhaseEstimator.out.1.bin')

    n = min(len(samples_orig), len(samples_filtered), len(samples_phase), len(samples_real))
    samples_orig = samples_orig[:n,0]  # Plot first channel
    samples_filtered = samples_filtered[:n,0]
    samples_phase = samples_phase[:n,0]
    samples_real = samples_real[:n,0]
    true_phase = true_phase[:n] if true_phase is not None else None
    true_inst_freq = true_inst_freq[:n] if true_inst_freq is not None else None

    filt_BW = f0 / 2
    l_freq = f0 - filt_BW / 2
    h_freq = f0 + filt_BW / 2

    cecht = ECHT(l_freq, h_freq, fs, filt_order=1, calibrate=True, f0=f0)
    cecht_Xf = cecht.fit_transform(samples_orig)
    cecht_phase = np.angle(cecht_Xf)

    echt = ECHT(l_freq, h_freq, fs, filt_order=1)
    echt_Xf = echt.fit_transform(samples_orig)
    echt_phase = np.angle(echt_Xf)

    hilbert_Xf = hilbert(samples_orig)
    hilbert_phase = np.angle(hilbert_Xf)

    if true_phase is not None:
        phase_error_deg = wrap_phase_deg(np.rad2deg(samples_phase - true_phase))
    else:
        phase_error_deg = wrap_phase_deg(np.rad2deg(samples_phase - hilbert_phase))

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "font.size": 11,
    })

    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.5, 1.3])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2])  # intentionally NOT sharing x

    ax0.plot(samples_orig, linewidth=1.2, label="Original")
    ax0.plot(samples_filtered, linewidth=1.2, label="Bandpass filtered")
    ax0.plot(samples_phase, linestyle="-.", linewidth=1.4, label="Online phase")
    ax0.plot(hilbert_phase, linewidth=1.2, alpha=0.7, label="Hilbert offline")
    if true_phase is not None:
        ax0.plot(true_phase, linewidth=1.2, alpha= 0.7, label="True phase (simulated)")
    # ax0.plot(cecht_phase, linestyle=":", linewidth=1.2, label="cecHT offline")
    # ax0.plot(echt_phase, linestyle="--", linewidth=1.2, label="ecHT offline")
    # ax0.plot(samples_real, linewidth=1.2, label="Online real part")

    ax0.set_ylabel("Amplitude / Phase (rad)")
    ax0.legend(frameon=False, ncol=3)

    ax1.plot(phase_error_deg, linewidth=1.2, label="Phase error")
    ax1.set_xlabel("Message index")
    ax1.set_ylabel("Error (deg)")
    ax1.legend(frameon=False)

    ax2.hist(phase_error_deg, bins=360*2, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.axvline(np.mean(phase_error_deg), linestyle="--", linewidth=1.2, label=f"Mean = {np.mean(phase_error_deg):.2f}°")
    ax2.axvline(np.median(phase_error_deg), linestyle=":", linewidth=1.2, label=f"Median = {np.median(phase_error_deg):.2f}°")
    ax2.set_xlabel("Phase Error (deg)")
    ax2.set_ylabel("Count")
    # ax2.set_xlim(-20, +20)
    ax2.legend(frameon=False)

    fig.savefig("samples_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_results(100, 8, "SourceClient")