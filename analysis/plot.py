"""
Plotting and EDF export.
"""

import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd
from scipy.signal import spectrogram
import mne

from .stimulus import get_edges

COMMON_BBOX = dict(
    facecolor="white",
    edgecolor="0.8",
    boxstyle="round,pad=0.2",
    alpha=0.85,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_polar_axis(ax, r_max, radial_ticks):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=[""] * 8)
    ax.set_ylim(0, r_max)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([rf"{t:.0f}%" for t in radial_ticks])
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=1.0)
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_rlabel_position(90)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("center")
    r0 = r_max * 1.15
    ax.text(np.deg2rad(45),  r0, r"$+45^\circ$", ha="center", va="center")
    ax.text(np.deg2rad(-45), r0, r"$-45^\circ$", ha="center", va="center")


def _circ_stats(phi_rad):
    phi = np.asarray(phi_rad, float)
    if phi.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mu  = float(circmean(phi, high=np.pi, low=-np.pi))
    sd  = float(circstd(phi,  high=np.pi, low=-np.pi))
    plv = float(np.abs(np.mean(np.exp(1j * phi))))
    pli = float(np.abs(np.mean(np.sign(np.sin(phi)))))
    return mu, sd, plv, pli


# ---------------------------------------------------------------------------
# Error plots
# ---------------------------------------------------------------------------

def plot_errors(errors: list[dict], time_range: tuple = None) -> None:
    """Plot error time series and polar histograms; save to output_path."""
    if not errors:
        print("No errors to plot.")
        return

    fig_time      = plt.figure(figsize=(10, 5))
    ax_ts         = fig_time.add_subplot(111)
    fig_polars = plt.figure(figsize=(3 * len(errors), 3.5))
    fig_polars.suptitle("Error distributions", fontsize=12)
    ax_polars  = [fig_polars.add_subplot(1, len(errors), i + 1, projection="polar")
                  for i in range(len(errors))]

    colors        = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bin_width_deg = 10
    bin_width_rad = np.radians(bin_width_deg)
    bins          = np.arange(-180, 181, bin_width_deg)
    centers       = (np.radians(bins[:-1]) + np.radians(bins[1:])) / 2

    hists = []
    for err in errors:
        vals = err["values"][~np.isnan(err["values"])]
        hists.append(np.histogram(vals, bins=bins)[0] / max(1, vals.size) * 100)

    r_max   = max(h.max() for h in hists) * 1.05
    r_max   = max(r_max, 5)
    r_ticks = [t for t in [10, 20, 30] if t < r_max]

    for i, err in enumerate(errors):
        color = colors[i % len(colors)]
        ls    = err.get("linestyle", "-")
        vals  = err["values"]

        ax_ts.plot(err["time_s"], vals, linestyle=ls, linewidth=1.2, color=color,
                   label=f"{err['label']} ({err['unit']})", alpha=0.85)

        if np.nansum(np.abs(vals)) == 0:
            continue
        vals_rad = np.radians(vals[~np.isnan(vals)])
        mu_u, sd_u, _, _ = _circ_stats(vals_rad)

        ax_polars[i].bar(centers, hists[i], width=bin_width_rad,
                         color=color, edgecolor="0", linewidth=0.75)
        ax_polars[i].plot([mu_u, mu_u], [0, r_max], color="0", linewidth=2)
        _style_polar_axis(ax_polars[i], r_max, r_ticks)
        ax_polars[i].text(
            0.5, 0.45,
            rf"${np.round(np.degrees(mu_u), 1) + 0.0:.1f}^\circ"
            rf" \pm {np.round(np.degrees(sd_u), 1):.1f}^\circ$",
            transform=ax_polars[i].transAxes, bbox=COMMON_BBOX, va="top", ha="center",
        )
        ax_polars[i].set_title(err["label"], fontsize=10)

    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("Error")
    ax_ts.legend(frameon=True, fontsize=8, loc="upper right")
    ax_ts.set_title("Error over time")
    if time_range is not None:
        ax_ts.set_xlim(time_range)

    fig_time.savefig("error_timeseries.png", dpi=300, bbox_inches="tight")
    fig_polars.savefig("error_distributions.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# IAF time series
# ---------------------------------------------------------------------------

def plot_iaf(ground_truth: dict, iaf_continuous: np.ndarray, samples: dict, start_ts: float, output_path: str="iaf_timeseries.png") -> None:
    """Plot estimated IAF over time with true IAF overlaid; adds error subplot when truth is known."""
    has_estimated = samples.get("IAFEstimator") is not None
    has_true      = ground_truth.get("true_inst_freq") is not None
    if not has_estimated and not has_true:
        return

    show_error = has_estimated and has_true
    nrows      = 2 if show_error else 1
    fig, axes  = plt.subplots(nrows, 1, sharex=True,
                              figsize=(10, 5 if show_error else 3),
                              height_ratios=[2, 1] if show_error else [1])
    ax     = axes[0] if show_error else axes
    ax_err = axes[1] if show_error else None
    ax_r = None

    time_s = (ground_truth["time"] - start_ts) / 1e6
    if has_true:
        ax.plot(time_s, ground_truth["true_inst_freq"],
                linewidth=1.5, alpha=0.8, label="True IAF")
        if ground_truth.get("true_amplitude") is not None:
            ax_r   = ax.twinx()
            ax_r.plot(time_s, ground_truth["true_amplitude"],
                      linestyle="--", linewidth=1.2, alpha=0.6, color="tab:gray", label="True amplitude")
            ax_r.set_ylabel("Amplitude", color="tab:gray")
    else:
        ax.plot(time_s, iaf_continuous, linewidth=1.5, alpha=0.8, label="Estimated IAF (Hilbert)")

    if has_estimated:
        x_est   = (samples["IAFEstimator"]["x"] - start_ts) / 1e6
        iaf_est = samples["IAFEstimator"]["y"]
        ax.plot(x_est, iaf_est, "o-", markersize=2.5, linewidth=1.2, label="Estimated IAF")

        if show_error:
            true_iaf = ground_truth["true_inst_freq"]
            n        = min(len(iaf_est), len(true_iaf))
            error    = iaf_est[:n] - true_iaf[:n]
            ax_err.plot(x_est[:n], error, linewidth=1.2, color="tab:red", label="IAF error")
            ax_err.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
            ax_err.axhline(np.nanmean(error), color="tab:red", linewidth=1.0, linestyle=":",
                           label=f"Mean = {np.nanmean(error):.3f} Hz")
            ax_err.set_ylabel("Error (Hz)")
            ax_err.set_xlabel("Time (s)")
            ax_err.legend(frameon=True, fontsize=8)
            ax_err.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

    ax.set_ylabel("Frequency (Hz)")
    if not show_error:
        ax.set_xlabel("Time (s)")

    ax.set_title("Individual Alpha Frequency")
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

    handles, labels     = ax.get_legend_handles_labels()
    handles_r, labels_r = ax_r.get_legend_handles_labels() if ax_r is not None else ([], [])
    ax.legend(handles + handles_r, labels + labels_r, frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Spectrum and time-series plots
# ---------------------------------------------------------------------------

def plot_spectrum(raw: np.ndarray, samples: dict, fs: float) -> None:
    """Plot FFT magnitude spectrum of raw and filtered signals."""
    freqs = np.fft.rfftfreq(len(raw), d=1 / fs)

    plt.figure(figsize=(8, 4))
    X = np.abs(np.fft.rfft(raw))
    plt.plot(freqs, X, label="Raw", alpha=0.7, color="blue")

    if samples.get("ecHTFilter") is not None:
        filt        = samples["ecHTFilter"]["y"]
        freqs_filt  = np.fft.rfftfreq(len(filt), d=1 / fs)
        plt.plot(freqs_filt, np.abs(np.fft.rfft(filt)),
                 label="Filtered", alpha=0.7, color="orange")

    plt.xlim(0, 20)
    plt.ylim(0, X[np.argmin(np.abs(freqs-0.5))])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Spectrum")
    plt.legend(facecolor="white", frameon=True)
    
    # plt.figure(figsize=(8, 4))
    # f, t, Sxx = spectrogram(raw, fs=fs, nperseg=int(fs*10))
    # plt.pcolormesh(t, f, np.log(Sxx), shading='nearest')
    # plt.colorbar(label='Intensity (log scale)')
    # plt.ylim(0, 50)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')


def plot_time_series(
    ground_truth: dict,
    samples: dict,
    hilbert_phase: np.ndarray,
    stim_ref: np.ndarray,
    time_range: tuple = None,
) -> None:
    time_s = (ground_truth["time"] - ground_truth["time"][0]) / 1e6
    n      = len(time_s)

    def _pick(label):
        d = samples.get(label)
        return d["y"] if d else None

    raw_eeg      = ground_truth["raw"] - np.mean(ground_truth["raw"])
    filt_online  = _pick("ecHTFilter")
    online_phase = _pick("PhaseEstimator_phase")
    true_phase   = ground_truth.get("true_phase")
    stimulus     = _pick("StimulusController")
    trigger      = _pick("SourceClient_TRIGGER")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 9), height_ratios=[3, 1])
    ax_raw, ax_phase = axes

    ax_raw.plot(time_s, raw_eeg[:n], color="0.6", linewidth=0.8, label="Raw EEG")
    if filt_online is not None:
        ax_raw.plot(time_s, filt_online[:n], color="0.3", linewidth=1.5, label="Filtered EEG (online)")
    if stim_ref is not None:
        ax_raw.fill_between(time_s, 0, 1, where=stim_ref[:n],
                            color="tab:blue", alpha=0.4,
                            transform=ax_raw.get_xaxis_transform(), label="Target stim")
    if trigger is not None:
        ax_raw.fill_between(time_s, 0, 1, where=trigger[:n],
                            color="tab:orange", alpha=0.4,
                            transform=ax_raw.get_xaxis_transform(), label="Trigger Stim")
    elif stimulus is not None:
        ax_raw.fill_between(time_s, 0, 1, where=stimulus[:n],
                            color="tab:orange", alpha=0.4,
                            transform=ax_raw.get_xaxis_transform(), label="Stimulus")

    ax_raw.set_ylabel("Amplitude (µV)")
    ax_raw.set_title("Raw and Filtered EEG with Stimuli")
    ax_raw.legend(facecolor="white", frameon=True, fontsize=8, loc="upper right")

    if hilbert_phase is not None:
        ax_phase.plot(time_s[:len(hilbert_phase)], hilbert_phase[:n],
                      color="tab:blue", linewidth=1.5, label="Offline Hilbert phase")
    if online_phase is not None:
        ax_phase.plot(time_s[:len(online_phase)], online_phase[:n],
                      color="tab:orange", linewidth=1.5, label="Online phase")
    if true_phase is not None:
        ax_phase.plot(time_s[:len(true_phase)], true_phase[:n],
                      color="tab:brown", linewidth=1.0, label="True phase")

    ax_phase.set_ylabel("Phase (rad)")
    ax_phase.set_title("Phase Estimates")
    ax_phase.legend(facecolor="white", frameon=True, fontsize=8, loc="upper right")
    ax_phase.set_xlabel("Time (s)")

    if time_range is not None:
        ax_raw.set_xlim(time_range)
    for ax in axes:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

    plt.savefig("time_series.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# EDF export
# ---------------------------------------------------------------------------

def write_edf(
    filepath: str,
    fs: float,
    ground_truth: dict,
    samples: dict,
    hilbert_phase: np.ndarray = None,
    stim_ref: np.ndarray = None,
    filtered: np.ndarray = None,
) -> None:
    """Write all pipeline signals to an EDF file."""
    raw  = ground_truth["raw"]
    time = ground_truth["time"]
    n    = len(raw)

    channels = [("Raw", "eeg", raw)]

    if filtered is not None:
        channels.append(("Filt_off", "misc", filtered[:n]))
    if samples.get("ecHTFilter") is not None:
        channels.append(("Filt_on",  "misc", samples["ecHTFilter"]["y"][:n]))
    if ground_truth.get("true_inst_freq") is not None:
        channels.append(("IAF_true_Hz", "misc", ground_truth["true_inst_freq"][:n]))
    if samples.get("IAFEstimator") is not None:
        channels.append(("IAF_est_Hz",  "misc", np.nan_to_num(samples["IAFEstimator"]["y"])[:n]))
    if hilbert_phase is not None:
        channels.append(("Hilbert_phi", "misc", hilbert_phase[:n]))
    if ground_truth.get("true_phase") is not None:
        channels.append(("True_phi",    "misc", ground_truth["true_phase"][:n]))
    if samples.get("PhaseEstimator_phase") is not None:
        phase_est = np.nan_to_num(samples["PhaseEstimator_phase"]["y"], nan=-2 * np.pi)
        channels.append(("Online_phi", "misc", phase_est[:n]))
    if stim_ref is not None:
        channels.append(("Target_Stim", "stim", stim_ref[:n]))
    if samples.get("StimulusController") is not None:
        channels.append(("Stimulus", "stim", samples["StimulusController"]["y"][:n]))
    if samples.get("SourceClient_TRIGGER") is not None:
        channels.append(("Trigger", "stim", samples["SourceClient_TRIGGER"]["y"][:n]))
    if samples.get("SourceClient_AUX") is not None:
        channels.append(("AUX", "misc", samples["SourceClient_AUX"]["y"][:n]))

    for key in sorted(k for k in samples if k.startswith("SourceClient_") and k.split("_")[1].isdigit()):
        ch_idx = int(key.split("_")[1])
        channels.append((f"EEG_{ch_idx + 1}", "eeg", samples[key]["y"][:n]))

    min_len  = min(len(ch[2]) for ch in channels)
    info     = mne.create_info([ch[0] for ch in channels], sfreq=fs,
                               ch_types=[ch[1] for ch in channels], verbose=False)
    raw_mne  = mne.io.RawArray([ch[2][:min_len] for ch in channels], info, verbose=False)
    raw_mne.apply_function(lambda x: x * 1e-6, picks="eeg")
    start_dt = datetime.datetime.fromtimestamp(
        time[0] / 1e6, tz=pytz.timezone("Europe/Berlin")
    ).replace(tzinfo=datetime.timezone.utc)
    raw_mne.set_meas_date(start_dt)
    raw_mne.export(filepath, fmt="edf", add_ch_type=True,
                   physical_range="channelwise", overwrite=True, verbose=False)
    print("EDF written.")


# ---------------------------------------------------------------------------
# ERP
# ---------------------------------------------------------------------------

def get_erp_windows(fs: float, samples: dict, channel: int = 1) -> list[dict]:
    """Extract EEG epochs around each trigger onset (-250 ms to +500 ms)."""
    ch = channel - 1  # zero-based

    if samples.get("ecHTFilter") is not None:
        eeg_y, eeg_t = samples["ecHTFilter"]["y"], samples["ecHTFilter"]["x"]
    elif samples.get(f"SourceClient_{ch}") is not None:
        eeg_y, eeg_t = samples[f"SourceClient_{ch}"]["y"], samples[f"SourceClient_{ch}"]["x"]
    elif samples.get(f"Producer_{ch}") is not None:
        eeg_y, eeg_t = samples[f"Producer_{ch}"]["y"], samples[f"Producer_{ch}"]["x"]
    else:
        print(f"Error: no EEG signal found for channel {channel}.")
        return []

    if samples.get("SourceClient_TRIGGER") is not None:
        trigger_y = samples["SourceClient_TRIGGER"]["y"]
    elif samples.get("StimulusController") is not None:
        trigger_y = samples["StimulusController"]["y"]
    else:
        print("Error: no trigger signal found.")
        return []

    time_s         = (eeg_t - eeg_t[0]) / 1e6
    trigger_binary = (trigger_y > 0.5).astype(float)
    onsets         = get_edges(trigger_binary, "rising")

    windows = []
    for idx in onsets:
        start_idx = idx - int(0.25 * fs)
        end_idx   = idx + int(0.5  * fs)
        if start_idx >= 0 and end_idx < len(eeg_y):
            windows.append({"time_s": time_s[start_idx:end_idx],
                            "eeg_y":  eeg_y[start_idx:end_idx]})
    return windows


def plot_erp_latency(windows: list[dict], fs: float) -> None:
    if not windows:
        print("No valid trigger windows found.")
        return

    print(f"Found {len(windows)} valid trigger windows")
    avg_eeg = np.mean([w["eeg_y"] for w in windows], axis=0)
    std_eeg = np.std( [w["eeg_y"] for w in windows], axis=0)
    time    = windows[0]["time_s"] - windows[0]["time_s"][int(0.25 * fs)]

    plt.figure(figsize=(8, 4))
    for w in windows[:20]:
        plt.plot(time, w["eeg_y"], color="0.8", linewidth=0.8, alpha=0.8)
    plt.plot(time, avg_eeg, color="tab:blue", linewidth=2, label="Average ERP")
    plt.fill_between(time, avg_eeg - std_eeg, avg_eeg + std_eeg,
                     color="tab:blue", alpha=0.3, label="±1 SD")
    plt.axvline(0, color="0.0", linestyle="--", label="Trigger onset")
    plt.xlabel("Peri-stimulus time (s)")
    plt.ylabel("EEG amplitude (µV)")
    plt.legend()
