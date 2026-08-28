"""
Plotting and EDF export.
"""

import csv
import datetime
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd
from scipy.signal import spectrogram, butter, sosfiltfilt, welch, periodogram, windows
import mne
import matplotlib as mpl

# Saving to a ".pgf" filename invokes the pgf backend automatically, so the
# default (interactive) backend stays active and plt.show() keeps working.
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

from .stimulus import get_edges

PLOTS_DIR = "/home/linda/Documents/MA/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

TEXTWIDTH    = 6.30045
ASPECT_RATIO = 9 / 16
SCALE        = 1.0
FIG_WIDTH    = TEXTWIDTH * SCALE
FIG_HEIGHT   = FIG_WIDTH * ASPECT_RATIO
FIGSIZE      = (FIG_WIDTH, FIG_HEIGHT)


def save_pgf(fig, name: str) -> None:
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.pgf"), bbox_inches="tight")

def save_png(fig, name: str) -> None:
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.png"), bbox_inches="tight", dpi=300)


COMMON_BBOX = dict(
    facecolor="white",
    edgecolor="0.8",
    boxstyle="round,pad=0.2",
    alpha=0.85,
)

CHANNEL_NAMES = {
    1: "Fp1",
    2: "Fpz",
    3: "Fz",
    4: "F3",
    5: "F7",
    27: "F4",
    29: "Fp2",
    28: "F8",
    9: "C3",
    25: "C4",
    6: "E1",
    30: "E2",
    10: "Cz",
    26: "FCz",
    12: "M1",
    23: "M2",
}

# E1/E2 are EOG (eye) channels, not scalp positions -- excluded from CSD/
# surface-Laplacian computations even though they're in CHANNEL_NAMES.
EOG_CHANNEL_NAMES = ("E1", "E2")


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

def plot_errors(errors: list[dict], plot_time=False, time_range: tuple = None,title=None) -> None:
    """Plot error time series and polar histograms; save to output_path."""
    if not errors:
        print("No errors to plot.")
        return

    if plot_time:
        fig_time      = plt.figure(figsize=FIGSIZE)
        fig_time.suptitle(f"Error time series - {title if title else ''}", fontsize=12)
        ax_ts         = fig_time.add_subplot(111)
        ax_ts_twin = ax_ts.twinx()  # Twin axis for amplitude if needed
    deg_indices  = [i for i, err in enumerate(errors) if err["unit"] == "degrees"]
    n_polar_rows = 2 if len(deg_indices) > 4 else 1
    n_polar_cols = int(np.ceil(len(deg_indices) / n_polar_rows)) if deg_indices else 1
    fig_polars = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT * n_polar_rows))
    fig_polars.suptitle(f"Error distributions {' - ' + title if title else ''}", fontsize=12)
    ax_polars  = {i: fig_polars.add_subplot(n_polar_rows, n_polar_cols, j + 1, projection="polar")
                  for j, i in enumerate(deg_indices)}

    colors        = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bin_width_deg = 10
    bin_width_rad = np.radians(bin_width_deg)
    bins          = np.arange(-180, 181, bin_width_deg)
    centers       = (np.radians(bins[:-1]) + np.radians(bins[1:])) / 2

    hists       = {}
    trimmed_rad = {}
    for i in deg_indices:
        vals = errors[i]["values"]
        vals = vals[~np.isnan(vals)][int(0.05*len(vals)):int(0.95*len(vals))]  # exclude first and last 20% to avoid edge effects
        hists[i]       = np.histogram(vals, bins=bins)[0] / max(1, vals.size) * 100
        trimmed_rad[i] = np.radians(vals)

    r_max   = max(h.max() for h in hists.values()) * 1.05 if hists else 5
    r_max   = max(r_max, 5)
    r_ticks = [t for t in [10, 20, 30] if t < r_max]

    for i, err in enumerate(errors):
        color = colors[i % len(colors)]
        ls    = err.get("linestyle", "-")
        vals  = err["values"]

        if plot_time:
            if err["unit"] == "degrees":
                ax_ts.plot(err["time_s"], vals, linestyle=ls, linewidth=1.2, color=color,
                    label=f"{err['label']} ({err['unit']})", alpha=0.85)
            else:
                ax_ts_twin.plot(err["time_s"], vals, linestyle=ls, linewidth=1.2, color=color,
                        label=f"{err['label']} ({err['unit']})", alpha=0.85)

        if err["unit"] != "degrees":
            continue
        if np.nansum(np.abs(vals)) == 0:
            continue
        mu_u, sd_u, _, _ = _circ_stats(trimmed_rad[i])

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

    if plot_time:
        # Take legend entries from both axes and combine them
        handles_ts, labels_ts = ax_ts.get_legend_handles_labels()
        handles_twin, labels_twin = ax_ts_twin.get_legend_handles_labels()
        ax_ts.set_xlabel("Time (s)")
        ax_ts.set_ylabel("Error")
        ax_ts_twin.set_ylabel("Error (Hz)")
        # Set one global legend with all entries
        ax_ts.legend(handles_ts + handles_twin, labels_ts + labels_twin, frameon=True, fontsize=8, loc="upper right")
        ax_ts.set_title("Error over time")
        if time_range is not None:
            ax_ts.set_xlim(time_range)

    # save_pgf(fig_time, "error_timeseries")
    # save_pgf(fig_polars, "error_distributions")
    if plot_time:
        save_png(fig_time, "error_timeseries")
    save_png(fig_polars, "error_distributions")



# ---------------------------------------------------------------------------
# IAF time series
# ---------------------------------------------------------------------------

def plot_iaf(ground_truth: dict, iaf_continuous: np.ndarray, samples: dict, start_ts: float, output_name: str = "iaf_timeseries") -> None:
    """Plot estimated IAF over time with true IAF overlaid; adds error subplot when truth is known."""
    has_estimated = samples.get("IAFEstimator") is not None
    has_true      = ground_truth.get("true_inst_freq") is not None
    if not has_estimated and not has_true:
        return

    show_error = has_estimated and has_true
    nrows      = 2 if show_error else 1
    fig, axes  = plt.subplots(nrows, 1, sharex=True,
                              figsize=FIGSIZE,
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
            ax_err.legend(loc="upper right")
            ax_err.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

    ax.set_ylabel("Frequency (Hz)")
    if not show_error:
        ax.set_xlabel("Time (s)")

    ax.set_title("Individual Alpha Frequency")
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

    handles, labels     = ax.get_legend_handles_labels()
    handles_r, labels_r = ax_r.get_legend_handles_labels() if ax_r is not None else ([], [])
    ax.legend(handles + handles_r, labels + labels_r, loc="upper right")

    ax.set_ylim(0,20)

    fig.tight_layout()
    save_pgf(fig, output_name)


# ---------------------------------------------------------------------------
# Spectrum and time-series plots
# ---------------------------------------------------------------------------

def plot_spectrum(raw: np.ndarray, samples: dict, filtered: np.ndarray, fs: float, aperiodic_params: tuple = None, X_white: np.ndarray = None) -> None:
    """Plot FFT magnitude spectrum of raw and filtered signals."""
    freqs = np.fft.rfftfreq(len(raw), d=1 / fs)
    slope, intercept = aperiodic_params

    # aperiodic_params were fit to a density-scaled PSD (Welch), where
    # Pxx(f) = 2*|X(f)|^2 / (fs*n). Rescale the intercept to raw FFT power
    # units: L(f) ~ |X(f)|^2. Take sqrt(L) to compare against the amplitude
    # spectrum plotted below (same convention as compute_hilbert_reference).
    n = len(raw)
    intercept_fft = intercept + np.log10(fs * n / 2)
    L = freqs[1:] ** slope * 10 ** intercept_fft  # skip f=0

    # plt.figure()
    # plt.plot(np.log10(freqs[1:]), np.log10(np.abs(np.fft.rfft(raw))[1:]), label="Raw", alpha=0.7, color="blue")
    # plt.plot(np.log10(freqs[1:]), np.log10(np.sqrt(L)), label="Aperiodic fit", alpha=0.7, color="red", linestyle="--")


    nrows = 2 if X_white is not None else 1
    fig, axes = plt.subplots(nrows, 1, figsize=FIGSIZE, squeeze=False, sharex=True)
    ax = axes[0, 0]
    X = np.abs(np.fft.rfft(raw))
    ax.plot(freqs, X, label="Raw", alpha=0.7, color="blue")

    if samples.get("ecHTFilter") is not None:
        filt        = samples["ecHTFilter"]["y"] #[300000:400000]
        taper = windows.tukey(len(filt), alpha=0.01)
        filt_wind = filt *taper
        freqs_filt  = np.fft.rfftfreq(len(filt), d=1 / fs)
        ax.plot(freqs_filt, np.abs(np.fft.rfft(filt_wind)), label="Filtered (online)", alpha=0.7, color="orange")

    ax.plot(freqs[1:], np.sqrt(L), label="Aperiodic fit", alpha=0.7, color="red", linestyle="--")

    if X_white is not None:
        ax2 = axes[1, 0]
        ax2.set_ylabel("Magnitude")
        ax2.plot(freqs, np.abs(X_white), label="Whitened", alpha=0.7, color="purple")
        ax2.plot(freqs, np.abs(np.fft.rfft(filtered)), label="Filtered (offline)", alpha=0.7, color="green")
        ax2.set_ylim(0, np.max(X_white[(freqs >= 5) & (freqs <= 10)]) * 5)
        ax2.legend(loc="upper right")
    else:
        ax.plot(freqs, np.abs(np.fft.rfft(filtered)), label="Filtered (offline)", alpha=0.7, color="green")

    ax.set_xlim(0, 100)
    # Use the maximum between 5-10Hz as xlim
    ax.set_ylim(0, np.max(X[(freqs >= 5) & (freqs <= 10)]) * 5)
    ax.set_ylabel("Magnitude")
    ax.set_title("Spectrum")
    (ax2 if X_white is not None else ax).set_xlabel("Frequency (Hz)")
    # get legend handler from both axis and combine them
    ax.legend(loc="upper right")

    # fig = plt.figure(figsize=FIGSIZE)
    # f, t, Sxx = spectrogram(raw, fs=fs, nperseg=int(fs*10))
    # plt.pcolormesh(t, f, np.log(Sxx), shading='nearest')
    # plt.colorbar(label='Intensity (log scale)')
    # plt.ylim(0, 50)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')

    save_pgf(fig, "spectrum")


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

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE, height_ratios=[3, 1])
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

    save_pgf(fig, "time_series")


# ---------------------------------------------------------------------------
# EDF export
# ---------------------------------------------------------------------------

def load_stim_annotations(results_dir: str, start_ts: float) -> "mne.Annotations | None":
    """Load stim event markers from a stim protocol log CSV in results_dir, if present.

    Each "on" or "off" row opens a segment that runs until the next logged row
    (whichever state that is), producing a "stim_on"/"stim_off" duration
    annotation. Non on/off rows (protocol_start/end/stopped) additionally get
    their own zero-duration marker.

    start_ts is the recording start time in the same units as ground_truth["time"]
    (microseconds since epoch), used to align log timestamps to the EDF timeline.
    """
    log_files = sorted(glob.glob(os.path.join(results_dir, "stim_protocol_log_*.csv")))
    if not log_files:
        return None

    with open(log_files[-1], newline="") as f:
        reader = csv.DictReader(f)
        rows = [(datetime.datetime.fromisoformat(row["timestamp"]).timestamp() - start_ts / 1e6,
                 row["state"]) for row in reader]

    onsets, durations, descriptions = [], [], []
    prev_time, prev_state = None, None
    for t, state in rows:
        if prev_state in ("on", "off"):
            onsets.append(prev_time)
            durations.append(t - prev_time)
            descriptions.append("stim_on" if prev_state == "on" else "stim_off")
        if state not in ("on", "off"):
            onsets.append(t)
            durations.append(0.0)
            descriptions.append(state)
        prev_time, prev_state = t, state

    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


# EDF/HDF5 export lives in analysis/edf_io.py (write_raw_signals_edf/load_runtime/
# write_analysis_edf) and analysis/runtime_meta.py (write_runtime_metadata/load_runtime_metadata).


# ---------------------------------------------------------------------------
# ERP
# ---------------------------------------------------------------------------

def apply_csd_transform(fs: float, samples: dict, channel: list[int]) -> dict[int, np.ndarray]:
    """Jointly re-reference every requested channel with a known scalp
    position (see CHANNEL_NAMES) using the CSD (surface Laplacian)
    transform, returning {channel: csd_y}.

    Channels without a known position, and EOG channels (E1/E2), are
    omitted -- callers should fall back to the as-recorded signal for those.
    """
    scalp_channels = [ch for ch in channel
                       if ch in CHANNEL_NAMES and CHANNEL_NAMES[ch] not in EOG_CHANNEL_NAMES]
    if not scalp_channels:
        return {}

    ys = []
    for ch in scalp_channels:
        ch0   = ch - 1
        entry = samples.get(f"SourceClient_{ch0}") or samples.get(f"Producer_{ch0}")
        ys.append(entry["y"])

    min_len = min(len(y) for y in ys)
    data    = np.array([y[:min_len] for y in ys]) * 1e-6  # uV -> V, mne's expected EEG unit

    info = mne.create_info([CHANNEL_NAMES[ch] for ch in scalp_channels], sfreq=fs,
                            ch_types="eeg", verbose=False)
    raw  = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1020")
    raw  = mne.preprocessing.compute_current_source_density(raw, copy=False)

    csd_data = raw.get_data() * 1e6  # back to uV, matching the untransformed samples
    return {ch: csd_data[i] for i, ch in enumerate(scalp_channels)}


def get_erp_windows(fs: float, samples: dict, channel: list[int] = [1],
                     bandpass: tuple[float, float] = (2.0, 30.0),
                     reject_uv: float = 10000.0, apply_csd: bool = False) -> list[dict]:
    """Extract EEG epochs around each trigger onset (-250 ms to +500 ms) for each channel.

    The EEG is bandpass-filtered (default 2-30 Hz) before epoching, and any
    epoch whose peak amplitude exceeds ``reject_uv`` (default ±100 µV) is
    discarded. Each returned window carries a 1-based "channel" key so
    callers can group windows by channel.

    If ``apply_csd`` is True, channels with a known scalp position are
    jointly re-referenced with a CSD (surface Laplacian) transform before
    epoching (see apply_csd_transform); channels without one (e.g. E1/E2)
    fall back to their as-recorded signal.
    """
    all_windows    = []
    csd_by_channel = apply_csd_transform(fs, samples, channel) if apply_csd else {}

    for ch in channel:
        ch0 = ch - 1  # zero-based

        if ch in csd_by_channel:
            entry = samples.get(f"SourceClient_{ch0}") or samples.get(f"Producer_{ch0}")
            eeg_y = csd_by_channel[ch]
            eeg_t = entry["x"][:len(eeg_y)]
        # if samples.get("ecHTFilter") is not None:
        #     eeg_y, eeg_t = samples["ecHTFilter"]["y"], samples["ecHTFilter"]["x"]
        elif samples.get(f"SourceClient_{ch0}") is not None:
            eeg_y, eeg_t = samples[f"SourceClient_{ch0}"]["y"], samples[f"SourceClient_{ch0}"]["x"]
        elif samples.get(f"Producer_{ch0}") is not None:
            eeg_y, eeg_t = samples[f"Producer_{ch0}"]["y"], samples[f"Producer_{ch0}"]["x"]
        else:
            print(f"Error: no EEG signal found for channel {ch}.")
            continue

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
        offsets        = get_edges(trigger_binary, "falling")

        # Bandpass-filter the continuous EEG before epoching, so filter edge
        # artifacts don't contaminate the epoch boundaries.
        sos   = butter(1, bandpass, btype="band", fs=fs, output="sos")
        eeg_y = sosfiltfilt(sos, eeg_y)

        n_rejected = 0
        for idx in onsets:
            start_idx = idx - int(0.25 * fs)
            end_idx   = idx + int(0.5  * fs)
            if start_idx >= 0 and end_idx < len(eeg_y):
                epoch = eeg_y[start_idx:end_idx]
                if np.max(np.abs(epoch)) > reject_uv:
                    n_rejected += 1
                    continue
                later_offsets = offsets[offsets > idx]
                stim_dur_s    = (time_s[later_offsets[0]] - time_s[idx]) if len(later_offsets) else None
                all_windows.append({"channel":    ch,
                                    "time_s":     time_s[start_idx:end_idx],
                                    "eeg_y":       epoch,
                                    "stim_dur_s":  stim_dur_s})

        if n_rejected:
            print(f"Channel {ch}: rejected {n_rejected} epoch(s) with peak amplitude > ±{reject_uv:.0f} µV.")

    return all_windows


def plot_erp_latency(windows: list[dict], fs: float) -> float | None:
    """Plot per-channel averaged ERPs stacked vertically on a shared x-axis.

    Windows are grouped by their "channel" key (windows without one, e.g.
    from older callers, are grouped together). The reported P1 latency is
    the mean of the per-channel P1 latencies.
    """
    if not windows:
        print("No valid trigger windows found.")
        return None

    channel_order = list(dict.fromkeys(w.get("channel") for w in windows))
    time           = windows[0]["time_s"] - windows[0]["time_s"][int(0.25 * fs)]
    post_stim_mask = (time >= 0.02) & (time <= 0.100)
    window_idx     = np.where(post_stim_mask)[0]

    channels = []
    for ch in channel_order:
        ch_windows = [w for w in windows if w.get("channel") == ch]
        avg_eeg    = np.mean([w["eeg_y"] for w in ch_windows], axis=0)
        std_eeg    = np.std( [w["eeg_y"] for w in ch_windows], axis=0)
        p1_idx     = window_idx[np.argmax(avg_eeg[window_idx])]
        p1_latency = time[p1_idx]
        channels.append({"channel": ch, "windows": ch_windows,
                         "avg": avg_eeg, "std": std_eeg, "p1_latency": p1_latency})
        label = f"channel {ch}" if ch is not None else "all channels"
        print(f"Found {len(ch_windows)} valid trigger windows for {label}; "
              f"P1 latency {p1_latency * 1000:.1f} ms")

    p1_latency = float(np.mean([c["p1_latency"] for c in channels]))
    print(f"Mean P1 latency across {len(channels)} channel(s): {p1_latency * 1000:.1f} ms")

    stim_durs = [w["stim_dur_s"] for w in windows if w["stim_dur_s"] is not None]
    stim_dur  = np.mean(stim_durs) if stim_durs else None
    if stim_dur is not None:
        print(f"Estimated stimulus duration: {stim_dur * 1000:.1f} ms")

    # Vertical spacing between channels, large enough that ±1 SD bands don't overlap.
    # span        = max(np.max(c["avg"] + c["std"]) - np.min(c["avg"] - c["std"]) for c in channels)
    span        = max(np.max(c["avg"]) - np.min(c["avg"]) for c in channels)
    offset_step = span * 1.2

    fig = plt.figure(figsize=FIGSIZE)
    if stim_dur is not None:
        plt.axvspan(0, stim_dur, color="0.6", alpha=0.2, zorder=0, label="Stimulus")

    yticks, yticklabels = [], []
    for i, c in enumerate(channels):
        offset = -i * offset_step
        # for w in c["windows"][:20]:
        #     plt.plot(time, w["eeg_y"] + offset, color="0.8", linewidth=0.8, alpha=0.8)
        plt.plot(time, c["avg"] + offset, color="tab:blue", linewidth=2,
                 label="Average ERP" if i == 0 else None)
        plt.fill_between(time, c["avg"] - c["std"] + offset, c["avg"] + c["std"] + offset,
                         color="tab:blue", alpha=0.2, label="±1 SD" if i == 0 else None)
        yticks.append(offset)
        yticklabels.append(f"{CHANNEL_NAMES.get(c['channel'], c['channel'])}" if c["channel"] is not None else "")

    plt.axvline(0, color="0.0", linestyle="--", label="Trigger onset")
    plt.axvline(p1_latency, color="tab:red", linestyle="--",
               label=f"Mean P1 latency ({p1_latency * 1000:.1f} ms)")

    plt.xlabel("Peri-stimulus time (s)")
    plt.ylabel("Channel" if len(channels) > 1 else "EEG amplitude (µV)")
    if len(channels) > 1:
        plt.yticks(yticks, yticklabels)
    plt.legend(loc="upper right")

    save_pgf(fig, "erp_latency")

    # Second figure: grand average ERP pooling every window from every channel.
    avg_pooled = np.mean([w["eeg_y"] for w in windows], axis=0)
    std_pooled = np.std( [w["eeg_y"] for w in windows], axis=0)
    p1_idx_pooled = window_idx[np.argmax(avg_pooled[window_idx])]
    p1_latency_pooled = time[p1_idx_pooled]
    print(f"Pooled P1 latency ({len(windows)} windows across {len(channels)} channel(s)): "
          f"{p1_latency_pooled * 1000:.1f} ms")

    fig_pooled = plt.figure(figsize=FIGSIZE)
    if stim_dur is not None:
        plt.axvspan(0, stim_dur, color="0.6", alpha=0.2, zorder=0, label="Stimulus")
    for w in windows[:20]:
        plt.plot(time, w["eeg_y"], color="0.8", linewidth=0.8, alpha=0.8)
    plt.plot(time, avg_pooled, color="tab:blue", linewidth=2, label="Average ERP")
    plt.fill_between(time, avg_pooled - std_pooled, avg_pooled + std_pooled,
                     color="tab:blue", alpha=0.2, label="±1 SD")
    plt.axvline(0, color="0.0", linestyle="--", label="Trigger onset")
    plt.axvline(p1_latency_pooled, color="tab:red", linestyle="--",
               label=f"P1 latency ({p1_latency_pooled * 1000:.1f} ms)")
    plt.xlabel("Peri-stimulus time (s)")
    plt.ylabel("EEG amplitude (µV)")
    plt.title("Grand average ERP (all channels pooled)")
    plt.legend(loc="upper right")
    save_pgf(fig_pooled, "erp_latency_pooled")

    return p1_latency
