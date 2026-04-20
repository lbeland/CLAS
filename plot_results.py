import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meegkit.phase import ECHT
from scipy.signal import hilbert
from read_output import get_signal_data

def wrap_phase_deg(x_deg):
    return (x_deg + 180.0) % 360.0 - 180.0

def get_results_file(processor_name, slot=0):
    # Look for files matching the pattern "Serializer2.0_PhaseEstimator.out.0.bin"

    for filename in os.listdir("rt_c_results"):
        if filename.endswith(f"{slot}.bin"):
            if processor_name in filename:
                return os.path.join("rt_c_results", filename)
    return None

def plot_results(fs, f0, graph_name):

    if graph_name == "TurboLinkCLAS":
        processors = ["SourceClient", "BandpassFilter", "PhaseEstimator", "IAFEstimator"]
        start_idx = 0
    elif graph_name == "IAFtests":
        processors = ["Producer", "IAFEstimator"]
        start_idx = int(1 * fs)
    elif graph_name == "ecHTtests":
        processors = ["Producer","PhaseEstimator"]
        start_idx = int(0.2 * fs)

    samples = {}
    for processor in processors:
        if processor == "Producer":
            for slot in range(4):
                signal = get_signal_data(get_results_file(processor, slot))
                if signal is not None:
                    samples[f"{processor}_{slot}"] = signal
        elif processor == "PhaseEstimator":
            for slot in range(2):
                signal = get_signal_data(get_results_file(processor, slot))
                if signal is not None:
                    samples[f"{processor}_{slot}"] = signal
        else:
            signal = get_signal_data(get_results_file(processor, 0))
            if signal is not None:
                samples[f"{processor}_0"] = signal

    n = min([len(samples) for samples in samples.values()])
    samples = {key: value[start_idx:n] for key, value in samples.items()}
    
    if not "SourceClient_0" in samples and "Producer_0" not in samples:
        print("Error: Original signal from SourceClient or Producer not found. Cannot plot results.")
        return
    elif "SourceClient_0" in samples:
        samples_orig = samples["SourceClient_0"]
        # Load data from simulate_client.py
        if os.path.exists("simulated_signal.npy"):
            loaded = np.load("simulated_signal.npy")
            assert samples_orig[0,0] == loaded["value"][0], "Loaded simulated signal does not match original signal from file"
            true_amplitude = loaded["amplitude"][start_idx:n]
            true_phase = np.angle(np.exp(1j * loaded["phase"][start_idx:n]))   # wrap to [-pi, +pi]
            true_inst_freq = loaded["inst_freq"][start_idx:n]
        else:
            true_amplitude = None
            true_phase = None
            true_inst_freq = None
    else:
        samples_orig = samples["Producer_0"]
        true_amplitude = samples["Producer_1"]
        true_phase = np.angle(np.exp(1j * samples["Producer_2"]))   # wrap to [-pi, +pi]
        true_inst_freq = samples["Producer_3"]

    # filt_BW = f0 / 2
    # l_freq = f0 - filt_BW / 2
    # h_freq = f0 + filt_BW / 2

    # cecht = ECHT(l_freq, h_freq, fs, filt_order=1, calibrate=True, f0=f0)
    # cecht_Xf = cecht.fit_transform(samples_orig)
    # cecht_phase = np.angle(cecht_Xf)

    # echt = ECHT(l_freq, h_freq, fs, filt_order=1)
    # echt_Xf = echt.fit_transform(samples_orig)
    # echt_phase = np.angle(echt_Xf)

    hilbert_Xf = hilbert(samples_orig)
    hilbert_phase = np.angle(hilbert_Xf)

    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    gs = fig.add_gridspec(4, 1) #, height_ratios=[2.2, 1.5, 1.3])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3])  # intentionally NOT sharing x

    ## Samples plot ----------------------
    ax0.plot(samples_orig, linewidth=1.2, label="Original")
    ax0.plot(hilbert_phase, linewidth=2, alpha=0.7, label="Hilbert offline")
    # ax0.plot(cecht_phase, linestyle=":", linewidth=1.2, label="cecHT offline")
    # ax0.plot(echt_phase, linestyle="--", linewidth=1.2, label="ecHT offline")
    if true_phase is not None:
        ax0.plot(true_phase, linewidth=1.2, alpha=0.7, label="True phase")
    if samples.get("BandpassFilter_0") is not None:
        ax0.plot(samples["BandpassFilter_0"], linewidth=1.2, label="Bandpass filtered")
    if samples.get("PhaseEstimator_0") is not None:
        samples_phase = samples["PhaseEstimator_0"]
        ax0.plot(samples_phase, linestyle="-.", linewidth=1.4, label="Online phase")
    # if samples.get("PhaseEstimator_1") is not None:
    #     samples_real = samples["PhaseEstimator_1"]
    #     ax0.plot(samples_real, linewidth=1.2, label="Online real part")

    ax0.set_ylabel("Amplitude / Phase (rad)")
    ax0.legend(frameon=False)

    ## Parameter plot ----------------------
    legend_items = []
    ax1b = ax1.twinx()
    if true_inst_freq is not None:
        line_true_freq = ax1.plot(true_inst_freq, label="True IAF", alpha=0.7)
        legend_items += line_true_freq
    if true_amplitude is not None:
        line_ampl = ax1b.plot(true_amplitude, label="True Amplitude", alpha=0.7)
        legend_items += line_ampl
    if samples.get("IAFEstimator_0") is not None:
        est_inst_freq = samples["IAFEstimator_0"]
        line_est_freq = ax1.plot(est_inst_freq, linewidth=1.2, label="Estimated IAF")
        legend_items += line_est_freq

    ax1.set_ylabel("Frequency (Hz)")
    ax1b.set_ylabel("Amplitude")
    ax1.legend(handles=legend_items, frameon=False)

    ## Error plot ----------------------
    units = []
    errors = []
    if samples.get("PhaseEstimator_0") is not None:
        samples_phase = samples["PhaseEstimator_0"]
        if true_phase is not None:
            error = np.angle(np.exp(1j * (true_phase - samples_phase)), deg=True)
        else:
            error = np.angle(np.exp(1j * (hilbert_phase - samples_phase)), deg=True)
        errors.append(error)
        ax2.plot(error, linewidth=1.2, label="Phase error")
        units.append("degrees")
    if samples.get("IAFEstimator_0") is not None and true_inst_freq is not None:
        error = est_inst_freq - true_inst_freq
        errors.append(error)
        ax2.plot(error, linewidth=1.2, label="Frequency error")
        units.append("Hz")

    ax2.set_xlabel("Message index")
    ax2.set_ylabel(f"Error ({" / ".join(units)})")
    ax2.tick_params(axis="y")


    ## Hist plot ----------------------
    for error in errors:
        ax3.hist(error, bins=360*2, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax3.axvline(np.mean(error), linestyle="--", linewidth=1.2, label=f"Mean = {np.mean(error):.2f}")
        ax3.axvline(np.median(error), linestyle=":", linewidth=1.2, label=f"Median = {np.median(error):.2f}")
        ax3.axvline(np.mean(error)+np.std(error), linestyle="-.", linewidth=1.2, label=f"Std = {np.std(error):.2f}")
        ax3.axvline(np.mean(error)-np.std(error), linestyle="-.", linewidth=1.2)

    ax3.set_xlabel("Error value")
    ax3.set_ylabel("Count")
    # ax3.set_xlim(-20, +20)
    ax3.legend(frameon=False)

    fig.savefig("samples_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_results(10000, 10, "ecHTtests")