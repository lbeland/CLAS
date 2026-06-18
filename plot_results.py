from multiprocessing import process
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from meegkit.phase import ECHT
from scipy.signal import hilbert, butter, sosfiltfilt, sosfreqz
from read_output import get_signal_data
from scipy.fft import fft, ifft, fftshift, ifftshift, next_fast_len
from pathlib import Path

RESULTS_DIR = "_last_run"

def wrap_phase_deg(x_deg):
    return (x_deg + 180.0) % 360.0 - 180.0

def _resolve_results_root(results_dir=None) -> Path:
    candidates = []
    if results_dir is not None:
        results_path = Path(results_dir)
        candidates.append(results_path)
        candidates.append(Path("results") / results_path)
    else:
        candidates.append(Path(RESULTS_DIR))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0]

def get_results_file(processor_name: str, name: str = ".out", slot: int = 0, results_dir=None) -> str | None:
    """Return path to the serializer output file for a given processor and slot."""
    root = _resolve_results_root(results_dir)
    if not root.exists():
        return None

    name_token = name.lstrip(".") if name else "out"
    expected_suffix = f"{processor_name}.{name_token}.{slot}.bin"

    matches = [path for path in root.rglob("*.bin") if path.name.endswith(expected_suffix)]
    if not matches:
        return None

    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(matches[0])

def plot_results(fs, f0, results_dir=RESULTS_DIR, timestamps=True):

    graph_file = glob.glob(os.path.join(results_dir, "*.yaml"))

    if len(graph_file) == 0:
        print(f"Error: no graph config (.yaml) found in {results_dir}")
        return

    graph_name = os.path.basename(graph_file[0])

    if graph_name == "TurboLinkCLAS.yaml" or graph_name == "ReplayCLAS.yaml":
        processors = ["SourceClient", "ecHTFilter", "PhaseEstimator", "IAFEstimator","StimulusController"] # "GlobalFilter",
    elif graph_name == "SimulateCLAS.yaml":
        processors = ["Producer", "ecHTFilter", "PhaseEstimator", "IAFEstimator", "StimulusController"]
    elif graph_name == "IAFtests.yaml":
        processors = ["Producer", "IAFEstimator"]
    elif graph_name == "ecHTtests.yaml":
        processors = ["Producer","PhaseEstimator"]

    samples = {}
    for processor in processors:
        if processor == "Producer":
            file = get_results_file(processor, slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(10)), timestamps=timestamps)
            if signal is None:
                continue
            time = time["hardware_ts"]
            for idx, channel in enumerate(signal.T):
                samples[f"{processor}_{idx}"] = {"x": time, "y": channel}
            # Metadata
            file = get_results_file(processor, name=".meta_out", slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(3)), timestamps=timestamps)
            if signal is None:
                continue
            time = time["hardware_ts"]
            for idx, channel in enumerate(signal.T):
                samples[f"{processor}_meta_{idx}"] = {"x": time, "y": channel}
        elif processor == "SourceClient":
            for slot in range(2):
                if slot == 0:
                    signal,time = get_signal_data(get_results_file(processor, slot=slot, results_dir=results_dir),channel = 0, timestamps=timestamps)
                    if signal is not None:
                        time = time["hardware_ts"]
                        samples[f"{processor}_{slot}"] = {"x": time, "y": signal}
                else:
                    channel = 0  # AUX channel
                    signal,time = get_signal_data(get_results_file(processor, slot=slot, results_dir=results_dir),channel = channel, timestamps=timestamps)
                    if signal is not None:
                        time = time["hardware_ts"]
                        samples[f"{processor}_{slot}_AUX"] = {"x": time, "y": signal}
                    channel = 8  # Trigger channel
                    signal,time = get_signal_data(get_results_file(processor, slot=slot, results_dir=results_dir),channel = channel, timestamps=timestamps)
                    if signal is not None:
                        time = time["hardware_ts"]
                        samples[f"{processor}_{slot}_TRIGGER"] = {"x": time, "y": signal}
        elif processor == "PhaseEstimator":
            for slot in range(2):
                signal, time = get_signal_data(get_results_file(processor, slot=slot, results_dir=results_dir), timestamps=timestamps)
                if signal is not None:
                    time = time["hardware_ts"]
                    samples[f"{processor}_{slot}"] = {"x": time, "y": signal}
        elif processor == "StimulusController":
            signal, time = get_signal_data(get_results_file(processor, slot=0, results_dir=results_dir), timestamps=timestamps)
            if signal is not None:
                time = time["hardware_ts"]
                samples[f"{processor}_0"] = {"x": time, "y": signal}
        else:
            signal, time = get_signal_data(get_results_file(processor, slot=0, results_dir=results_dir), timestamps=timestamps)
            if signal is not None:
                time = time["hardware_ts"]
                samples[f"{processor}_0"] = {"x": time, "y": signal}

    if not "SourceClient_0" in samples and "Producer_0" not in samples:
        print("Error: Original signal from SourceClient or Producer not found. Cannot plot results.")
        return
    elif "SourceClient_0" in samples:
        samples_orig = samples["SourceClient_0"]["y"]
        time_orig = samples["SourceClient_0"]["x"]
        # Load data from simulate_client.py
        if os.path.exists("simulated_signal.npy"):
            loaded = np.load("simulated_signal.npy")
            assert samples_orig[0,0] == loaded["value"][0], "Loaded simulated signal does not match original signal from file"
            true_amplitude = loaded["amplitude"]
            true_phase = np.angle(np.exp(1j * loaded["phase"]))   # wrap to [-pi, +pi]
            true_inst_freq = loaded["inst_freq"]
        else:
            true_amplitude = None
            true_phase = None
            true_inst_freq = None
    else:
        samples_orig = samples["Producer_0"]["y"]
        time_orig = samples["Producer_0"]["x"]
        true_amplitude = samples["Producer_meta_0"]["y"]
        true_phase = np.angle(np.exp(1j * samples["Producer_meta_1"]["y"]))   # wrap to [-pi, +pi]
        true_inst_freq = samples["Producer_meta_2"]["y"]

    # filt_BW = f0 / 2
    # l_freq = f0 - filt_BW / 2
    # h_freq = f0 + filt_BW / 2

    # cecht = ECHT(l_freq, h_freq, fs, filt_order=1, calibrate=True, f0=f0)
    # cecht_Xf = cecht.fit_transform(samples_orig)[:,0]
    # cecht_phase = np.angle(cecht_Xf)

    # echt = ECHT(l_freq, h_freq, fs, filt_order=1)
    # echt_Xf = echt.fit_transform(samples_orig)[:,0]
    # echt_phase = np.angle(echt_Xf)

    sos = butter(1, [4, 8], btype='band', fs=fs, output='sos')
    samples_filt = sosfiltfilt(sos, samples_orig)
    hilbert_Xf = hilbert(samples_filt)
    hilbert_phase = np.angle(hilbert_Xf)

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    gs = fig.add_gridspec(4, 1) #, height_ratios=[2.2, 1.5, 1.3])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3])  # intentionally NOT sharing x

    # Add grid to all plots
    for ax in [ax0, ax1, ax2]:
        ax.minorticks_on()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

    # time = np.arange(start_idx, start_idx + len(samples_orig))

    ## Samples plot ----------------------
    start_ts = time_orig[0]
    time_orig = (time_orig - start_ts)/1e6   # align to 0
    ax0.plot(time_orig, samples_orig, 'o-',linewidth=1.2, markersize=2.5,label="Original")
    ax0b = ax0.twinx()
    ax0b.plot(time_orig, np.rad2deg(hilbert_phase), linewidth=2, alpha=0.7, label="Hilbert offline",c="tab:red")
    # ax0.plot(cecht_phase, linestyle=":", linewidth=1.2, label="cecHT offline")
    # ax0.plot(echt_phase, linestyle="--", linewidth=1.2, label="ecHT offline")
    if true_phase is not None:
        ax0b.plot(time_orig, np.rad2deg(true_phase), linewidth=1.2, alpha=0.7, label="True phase")
    if samples.get("GlobalFilter_0") is not None:
        x = samples["GlobalFilter_0"]["x"]
        ax0.plot((x-start_ts)/1e6, samples["GlobalFilter_0"]["y"], linewidth=1.2, label="Bandpass filtered")
    if samples.get("PhaseEstimator_0") is not None:
        x = samples["PhaseEstimator_0"]["x"]
        samples_phase = samples["PhaseEstimator_0"]["y"]
        ax0b.plot((x-start_ts)/1e6, np.rad2deg(samples_phase), linestyle="-.", linewidth=1.4, label="Online phase",c="tab:olive")
    if samples.get("StimulusController_0") is not None:
        x = samples["StimulusController_0"]["x"]
        samples_stim = np.where(samples["StimulusController_0"]["y"] == 1)[0]
        ax0.plot((x[samples_stim]-start_ts)/1e6, 0*samples["StimulusController_0"]["y"][samples_stim], "o", alpha=0.5,markersize=3, label="Stimulus")
        # ax0.plot(time_orig[samples_stim], samples_orig[samples_stim], "o", markersize=3, label="Stimulus")
    if samples.get("SourceClient_1_AUX") is not None:
        x = samples["SourceClient_1_AUX"]["x"]
        ax0.plot((x-start_ts)/1e6, samples["SourceClient_1_AUX"]["y"], linewidth=1.2, label="AUX")
    if samples.get("SourceClient_1_TRIGGER") is not None:
        x = samples["SourceClient_1_TRIGGER"]["x"]
        samples_stim = np.where(samples["SourceClient_1_TRIGGER"]["y"] == 1)[0]
        ax0.plot((x[samples_stim]-start_ts)/1e6, 0*samples["SourceClient_1_TRIGGER"]["y"][samples_stim], "o", alpha=0.5, markersize=3, label="Trigger")
    
    # if samples.get("PhaseEstimator_1") is not None:
    #     samples_real = samples["PhaseEstimator_1"]
    #     ax0.plot(time, samples_real, linewidth=1.2, label="Online real part")

    ax0.set_ylabel("Amplitude")
    ax0b.set_ylabel("Phase (degrees)")
    handles, labels = ax0.get_legend_handles_labels()
    handles2, labels2 = ax0b.get_legend_handles_labels()
    ax0.legend(handles + handles2, labels + labels2, loc='upper left', frameon=False)
    # ax0.legend(loc='upper left', frameon=False)

    ## Parameter plot ----------------------
    legend_items = []
    ax1b = ax1.twinx()
    if true_inst_freq is not None:
        line_true_freq = ax1.plot(time_orig, true_inst_freq, label="True IAF", alpha=0.7)
        legend_items += line_true_freq
    if true_amplitude is not None:
        line_ampl = ax1b.plot(time_orig, true_amplitude, '--', label="True Amplitude", alpha=0.7)
        legend_items += line_ampl
    if samples.get("IAFEstimator_0") is not None:
        x = samples["IAFEstimator_0"]["x"]
        est_inst_freq = samples["IAFEstimator_0"]["y"]
        line_est_freq = ax1.plot((x-start_ts)/1e6, est_inst_freq, 'o-', markersize=2.5,linewidth=1.2, label="Estimated IAF")
        legend_items += line_est_freq

    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.set_ylabel("Frequency (Hz)")
    ax1b.set_ylabel("Amplitude")
    ax1.legend(handles=legend_items, frameon=False)

    ## Error plot ----------------------
    # calibs = get_hilbert_calib(len(samples_orig), fs, f0)
    # calibs.tofile("hilbert_calibs.csv",sep=',')
    units = []
    errors = []
    if samples.get("PhaseEstimator_0") is not None:
        x = samples["PhaseEstimator_0"]["x"]
        samples_phase = samples["PhaseEstimator_0"]["y"]
        n = len(samples_phase)
        if true_phase is not None:
            error = np.angle(np.exp(1j * (true_phase - samples_phase)), deg=True)
            # cecht_error = np.angle(np.exp(1j * (true_phase - cecht_phase)), deg=True)
            # ax2.plot(message_indices, cecht_error, linestyle=":", linewidth=1.2, label="cecHT error")
            # echt_error = np.angle(np.exp(1j * (true_phase - echt_phase)), deg=True)
            # ax2.plot(message_indices, echt_error, linestyle="--", linewidth=1.2, label="ecHT error")
            hilbert_error = np.angle(np.exp(1j * (true_phase - hilbert_phase)), deg=True)
            # hlbert_error_calib = np.angle(np.exp(1j * (true_phase - (np.angle(hilbert_Xf*calibs)))), deg=True)
            ax2.plot((x-start_ts)/1e6, hilbert_error, linestyle="--", linewidth=1.2, label="Hilbert error")
            # ax2.plot((x-start_ts)/1e6, hlbert_error_calib, linestyle=":", linewidth=1.2, label="Hilbert error (calibrated)")
        else:
            n = min(len(samples_phase), len(hilbert_phase))
            error = np.angle(np.exp(1j * (hilbert_phase[:n] - samples_phase[:n])), deg=True)
        errors.append(error)
        ax2.plot((x-start_ts)/1e6, error, linewidth=1.2, label="Phase error")
        units.append("degrees")
    if samples.get("IAFEstimator_0") is not None and true_inst_freq is not None:
        x = samples["IAFEstimator_0"]["x"]
        error = est_inst_freq - true_inst_freq
        errors.append(error)
        ax2.plot((x-start_ts)/1e6, error, linewidth=1.2, label="Frequency error")
        units.append("Hz")
    ax2.set_ylabel(f"Error ({" / ".join(units)})")
    ax2.tick_params(axis="y")
    ax2.legend(frameon=False)
    # ax2.set_xlim(49900, 50000)
    # ax2.set_ylim(-1, 1)
    ax2.set_xlabel("Time (s)")


    ## Hist plot ----------------------
    for color, error in zip(['tab:orange', 'tab:green'], errors):
        if np.nansum(error) == 0:
            continue
        try:
            ax3.hist(error, bins="auto",alpha=0.8, edgecolor="black", linewidth=0.5, color=color)
        except ValueError:
            print(f"Unique error values: {np.unique(error)}")
        ax3.axvline(np.nanmean(error), linestyle="--", linewidth=1.2, label=f"Mean = {np.nanmean(error):.2f}", color=color)
        ax3.axvline(np.nanmedian(error), linestyle=":", linewidth=1.2, label=f"Median = {np.nanmedian(error):.2f}", color=color)
        ax3.axvline(np.nanmean(error)+np.nanstd(error), linestyle="-.", linewidth=1.2, label=f"Std = {np.nanstd(error):.2f}", color=color)
        ax3.axvline(np.nanmean(error)-np.nanstd(error), linestyle="-.", linewidth=1.2, color=color)

    ax3.set_xlabel("Error value")
    ax3.set_ylabel("Count")
    # ax3.set_xlim(-20, +20)
    ax3.legend(frameon=False)

    fig.savefig("samples_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_results(10000, 10.0, results_dir="_last_run")