"""
Draft analysis: log power change (Stim On vs Stim Off) per frequency bin,
grouped by stim_onset_deg condition.
    python -m analysis.spectrum_analysis   (from CLAS/)

Discovers run directories under results/CLAS*/*/, reads each run's graph
yaml to find its stim_onset_deg condition, loads raw_signals.edf (written
by analysis/edf_io.py:write_raw_signals_edf, via analysis/main.py), splits
the recording into "stim_on" / "stim_off" segments using the annotations
added by load_stim_annotations,
computes a Welch PSD per segment on one channel, and plots the log10(on /
off) power ratio per frequency bin (mean +/- SEM across Stim On trials)
for each stim_onset_deg condition.

This is a first draft (single channel, no clustering/stats yet) meant as a
starting point before adding multi-channel handling and statistics akin to
fig_2.m panels B/C.
"""

import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml
from scipy.signal import welch
from scipy.stats import ttest_ind, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.plot import FIGSIZE, FIG_WIDTH, FIG_HEIGHT, save_pgf, save_png, CHANNEL_NAMES, EOG_CHANNEL_NAMES

CONDITIONS      = ("stim_on", "stim_off")
STIM_ONSET_DEGS = (0, 60, 150, 180, 240, 330)

CONDITION_LABELS = {
    330: "Pre-Peak",
    180: "P1-Trough",
    60:  "Post-Peak",
    150: "Pre-Trough",
    0:   "P1-Peak",
    240: "Post-Trough",

}
CONDITION_COLORS = {
    0:   "green",  # P1-Peak
    60:  "red",
    150: "green",
    180: "blue",  # P1-Trough
    240: "orange",
    330: "blue",
}

# The two P1 conditions (stim onset at the peak/trough itself, vs. the other
# four which lead/lag the peak or trough) are drawn dashed so they stand out
# as a pair from the four solid phase-offset conditions.
CONDITION_LINESTYLES = {0: "--", 180: "--"}

# Opposite-phase comparisons for the time-frequency t-maps (fig_2.m panel B):
# Pre-Peak vs Pre-Trough, and Post-Peak vs Post-Trough.
TF_COMPARISONS = ((330, 150), (60, 240))

# Extended alpha band the per-condition significance test is restricted to.
EXTENDED_ALPHA_RANGE = (6, 14)


def find_edf_path(results_dir: str) -> str:
    """Resolve the raw-signals EDF file for a results directory (written by
    analysis/edf_io.py:write_raw_signals_edf, via analysis/main.py). All-channel
    EEG data lives only in raw_signals.edf -- analysis.edf only carries the
    single selected channel."""
    edf_path = Path(results_dir) / "raw_signals.edf"
    if not edf_path.exists():
        raise FileNotFoundError(f"No raw_signals.edf found in {results_dir}")
    return str(edf_path)


def discover_condition_runs(
    base_dir: str = "results",
    stim_onset_degs: tuple[int, ...] = STIM_ONSET_DEGS,
) -> dict[int, list[str]]:
    """Group run directories by their stim_onset_deg condition.

    Pools across every top-level CLAS*-prefixed session folder found under
    base_dir (e.g. multiple subjects/sessions), not just a single one.
    """
    runs_by_condition = {deg: [] for deg in stim_onset_degs}

    session_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, "CLAS*")) if os.path.isdir(d))
    if len(session_dirs) > 1:
        print(f"Pooling across {len(session_dirs)} CLAS* session folders: "
              f"{[os.path.basename(d) for d in session_dirs]}")

    for session_dir in session_dirs:
        for run_dir in sorted(glob.glob(os.path.join(session_dir, "*"))):
            if not os.path.isdir(run_dir):
                continue
            yaml_files = glob.glob(os.path.join(run_dir, "TurboLinkCLAS.yaml"))
            if not yaml_files:
                continue

            with open(yaml_files[0]) as f:
                graph_config = yaml.safe_load(f)
            stim_onset_deg = (
                graph_config.get("graph", {})
                .get("processors", {})
                .get("StimulusController", {})
                .get("options", {})
                .get("stim_onset_deg")
            )

            if stim_onset_deg in runs_by_condition:
                runs_by_condition[stim_onset_deg].append(run_dir)
            elif stim_onset_deg is not None:
                print(f"Skipping {run_dir}: stim_onset_deg={stim_onset_deg} not in {stim_onset_degs}")

    return runs_by_condition


def resolve_channel(raw: "mne.io.BaseRaw", name: str) -> str:
    """Look up a channel by name (raw_signals.edf/analysis.edf are written
    with add_ch_type=False, so labels carry no type prefix to strip)."""
    if name not in raw.ch_names:
        raise ValueError(f"Channel '{name}' not found in {raw.ch_names}")
    return name


def apply_csd_reference(raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
    """Replace the online (hardware) reference with a surface Laplacian /
    current source density (CSD) reference, computed across all channels
    with a known 10-20 position (see CHANNEL_NAMES) -- write_raw_signals_edf
    already labels those channels with their real electrode name (e.g. "Fz").

    CSD is reference-free by construction, so the recorded (hardware-
    referenced) signal is used directly -- no need to first undo the old
    online reference. Channels without a known position (AUX, Trigger, and
    any EEG_N channel with no mapped 10-20 site) are dropped, since
    compute_current_source_density requires every "eeg"-typed channel to
    have one. E1/E2 are relabelled as EOG so they're excluded from the CSD
    computation rather than dropped.

    Channels are identified by name, not by raw.get_channel_types(): the EDF
    round-trip (write_raw_signals_edf -> read_raw_edf) loses the original channel
    types, so every channel comes back typed "eeg" regardless of what it was
    exported as.
    """
    known = set(CHANNEL_NAMES.values())
    raw.drop_channels([ch for ch in raw.ch_names if ch not in known])

    scalp = known - set(EOG_CHANNEL_NAMES)
    raw.set_channel_types({
        ch: "eeg" if ch in scalp else "eog"
        for ch in raw.ch_names
    }, verbose=False)
    raw.set_montage("standard_1020")

    return mne.preprocessing.compute_current_source_density(raw, copy=False)


def get_condition_segments(raw: "mne.io.BaseRaw") -> dict[str, list[tuple[int, int]]]:
    """Return {"stim_on": [(start_idx, end_idx), ...], "stim_off": [...]} sample ranges from annotations."""
    segments  = {cond: [] for cond in CONDITIONS}
    n_samples = len(raw.times)

    for ann in raw.annotations:
        if ann["description"] not in CONDITIONS:
            continue
        start_idx = int(raw.time_as_index(ann["onset"])[0])
        end_idx   = min(int(raw.time_as_index(ann["onset"] + ann["duration"])[0]), n_samples)
        if end_idx > start_idx:
            segments[ann["description"]].append((start_idx, end_idx))

    return segments


def clean_and_filter_channel(
    raw: "mne.io.BaseRaw",
    channel: str,
    lowpass_hz: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowpass-filter one channel.

    Returns (cleaned_signal, retained_idx), where retained_idx[i] is the sample
    index in the original recording that cleaned_signal[i] came from - used to
    translate stim_on/stim_off sample ranges onto the cleaned timeline. Every
    sample is currently retained (retained_idx is a no-op identity mapping);
    kept so extract_segment()'s searchsorted-based translation still works if
    sample-dropping (e.g. online-IAF-based, from runtime_metadata.h5) is added
    back later.
    """
    fs     = raw.info["sfreq"]
    signal = raw.get_data(picks=resolve_channel(raw, channel))[0] * 1e6  # V -> uV

    retained_idx   = np.arange(len(signal))
    cleaned_signal = mne.filter.filter_data(signal, fs, l_freq=None, h_freq=lowpass_hz, verbose=False)

    return cleaned_signal, retained_idx


def extract_segment(
    cleaned_signal: np.ndarray,
    retained_idx: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """Slice the cleaned (concatenated) signal to the samples originally in [start_idx, end_idx)."""
    lo, hi = np.searchsorted(retained_idx, (start_idx, end_idx))
    return cleaned_signal[lo:hi]


@dataclass
class RunData:
    fs: float
    on_arrays: list[np.ndarray]
    off_arrays: list[np.ndarray]


def load_run(
    results_dir: str,
    channel: str = "Fz",
    lowpass_hz: float = 30.0,
) -> RunData | None:
    """Load, re-reference, clean, and epoch one run's EDF into Stim On/Off sample arrays.

    This is the expensive, shared preprocessing step (EDF read, CSD
    reference, IAF-based cleaning + lowpass filter) needed by both the
    per-condition spectrum plot and the time-frequency t-maps -- run it once
    per run directory and reuse the result for both.
    """
    edf_path = find_edf_path(results_dir)
    raw      = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    apply_csd_reference(raw)
    fs = raw.info["sfreq"]

    cleaned_signal, retained_idx = clean_and_filter_channel(raw, channel, lowpass_hz=lowpass_hz)
    segments = get_condition_segments(raw)

    on_arrays  = [extract_segment(cleaned_signal, retained_idx, s, e) for s, e in segments["stim_on"]]
    off_arrays = [extract_segment(cleaned_signal, retained_idx, s, e) for s, e in segments["stim_off"]]
    print(f"  {results_dir}: stim_on: {len(on_arrays)} segment(s), stim_off: {len(off_arrays)} segment(s)")

    if not on_arrays or not off_arrays:
        return None
    return RunData(fs=fs, on_arrays=on_arrays, off_arrays=off_arrays)


def compute_welch_psd(
    segment_arrays: list[np.ndarray],
    fs: float,
    win_s: float = 1.0,
    n_fft: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Welch PSD per segment array. Returns (freqs, psd[n_segments, n_freqs])."""
    nperseg = int(win_s * fs)

    freqs = None
    psds  = []
    for i, data in enumerate(segment_arrays):
        if len(data) < nperseg:
            print(f"Skipping segment {i}: shorter than window ({win_s}s) after cleaning")
            continue
        freqs, pxx = welch(data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hamming", nfft=n_fft)
        psds.append(pxx)

    return freqs, np.array(psds) if psds else np.empty((0, 0))


def compute_welch_spectrogram(
    data: np.ndarray,
    fs: float,
    win_s: float = 1.0,
    step_s: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Time-resolved Welch PSD via a sliding window (hop = step_s, window = win_s).

    Returns (freqs, times_s, psd) with psd of shape (n_times, n_freqs); times_s
    are seconds from the start of `data`, at the center of each window.
    """
    nperseg = int(win_s * fs)
    step    = int(step_s * fs)

    freqs = None
    times = []
    psds  = []
    start = 0
    while start + nperseg <= len(data):
        f, pxx = welch(data[start:start + nperseg], fs=fs,
                        nperseg=nperseg, window="hamming")
        freqs = f
        psds.append(pxx)
        times.append((start + nperseg / 2) / fs)
        start += step

    return freqs, np.array(times), np.array(psds) if psds else np.empty((0, 0))


def movmean(x: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """Centered moving average with `window // 2` points on each side (edges shrink, like MATLAB movmean)."""
    half = window // 2
    x    = np.moveaxis(x, axis, 0)
    out  = np.empty_like(x, dtype=float)
    for i in range(x.shape[0]):
        lo, hi   = max(0, i - half), min(x.shape[0], i + half + 1)
        out[i] = x[lo:hi].mean(axis=0)
    return np.moveaxis(out, 0, axis)


def compute_trial_tf_log_power_change(
    on_data: np.ndarray,
    baseline_psd: np.ndarray,
    fs: float,
    win_s: float = 1.0,
    step_s: float = 1.0,
    smooth_window_s: float = 10.0,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Time-resolved log10(on / off-baseline) power change for one Stim On trial.

    Smoothed across time with a movmean window of smooth_window_s seconds
    (5 preceding + 5 following, per the paper's methods).
    Returns (freqs, times_s, log_power_change) with shape (n_times, n_freqs).
    """
    freqs, times, psd = compute_welch_spectrogram(on_data, fs, win_s=win_s, step_s=step_s)
    if psd.size == 0:
        return freqs, times, psd

    log_power_change = np.log10(psd / baseline_psd)
    smooth_window     = int(round(smooth_window_s / step_s))
    log_power_change  = movmean(log_power_change, smooth_window, axis=0)
    return freqs, times, log_power_change


def compute_log_power_change(psd_on: np.ndarray, psd_off: np.ndarray) -> np.ndarray:
    """log10(on / off) power ratio per frequency bin, per Stim On trial.

    Uses the mean Stim Off spectrum as baseline, so trial counts need not match.
    Returns an array of shape (n_on_trials, n_freqs).
    """
    baseline = psd_off.mean(axis=0)
    return np.log10(psd_on / baseline)


def compute_condition_ttest(
    freqs: np.ndarray,
    log_ratio: np.ndarray,
    freq_range: tuple = EXTENDED_ALPHA_RANGE,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-sample t-test of log10(on/off) power ratio against 0 (no change),
    per frequency bin, restricted to freq_range.

    Returns (freqs_band, t_stat, significant), each of shape (n_freqs_in_band,);
    significant is the boolean mask p < alpha.
    """
    in_band    = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_band = freqs[in_band]
    t_stat, p_value = ttest_1samp(log_ratio[:, in_band], popmean=0, axis=0)
    return freqs_band, t_stat, p_value < alpha


def plot_log_power_change(
    freqs: np.ndarray,
    log_ratio: np.ndarray,
    channel: str,
    xlim: tuple = (4, 13),
) -> None:
    """Plot mean +/- SEM log10(on / off) power ratio per frequency bin."""
    fig  = plt.figure(figsize=FIGSIZE)
    mean = log_ratio.mean(axis=0)
    sem  = log_ratio.std(axis=0) / np.sqrt(log_ratio.shape[0])

    plt.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    plt.plot(freqs, mean, color="tab:purple", label=f"log10(on/off) (n={log_ratio.shape[0]})")
    plt.fill_between(freqs, mean - sem, mean + sem, color="tab:purple", alpha=0.3)

    plt.xlim(xlim)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(f"Power change ({channel}, log10 on/off)")
    plt.title("Stim On vs Stim Off log power change")
    plt.legend(loc="upper left", frameon=True)

    save_pgf(fig, "stim_on_off_log_power_change")


def plot_condition_log_power_change(
    freqs: np.ndarray,
    log_ratio_by_condition: dict[int, np.ndarray],
    channel: str,
    xlim: tuple = (4, 17),
    alpha_range: tuple = EXTENDED_ALPHA_RANGE,
) -> None:
    """Plot mean log10(on / off) power ratio per stim_onset_deg condition, with a
    significance track below marking the frequencies (within alpha_range) where a
    condition's power change differs significantly from zero (one-sample t-test,
    p < 0.05, uncorrected)."""
    conditions  = [c for c in CONDITION_LABELS if c in log_ratio_by_condition]
    row_spacing = 0.1  # was 1.0 -- distance between row centers, packs the bars together
    bar_height  = 0.09  # was 0.8 -- leaves a thin gap between rows at this spacing
    fig, (ax, ax_sig) = plt.subplots(
        2, 1, figsize=(FIG_WIDTH*1.3, FIG_WIDTH), sharex=True,
        gridspec_kw={"height_ratios": [8, 1], "hspace": 0},
    )

    ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    freq_step = np.median(np.diff(freqs))
    for row, condition in enumerate(conditions):
        log_ratio = log_ratio_by_condition[condition]
        color     = CONDITION_COLORS.get(condition, "0.3")
        linestyle = CONDITION_LINESTYLES.get(condition, "-")
        label     = CONDITION_LABELS.get(condition, f"{condition}°")
        mean = log_ratio.mean(axis=0)
        # sem  = log_ratio.std(axis=0) / np.sqrt(log_ratio.shape[0])
        ax.plot(freqs, mean, color=color, linestyle=linestyle, label=f"{label}")# (n={log_ratio.shape[0]})")
        # ax.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.2)

        freqs_band, _, significant = compute_condition_ttest(freqs, log_ratio, freq_range=alpha_range,alpha=0.05/len(conditions))
        # Drop bins whose bar would extend past alpha_range rather than truncating them.
        fully_in_range = (freqs_band - freq_step / 2 >= alpha_range[0]) & (freqs_band + freq_step / 2 <= alpha_range[1])
        significant &= fully_in_range
        bottom = row * row_spacing
        if linestyle == "--":
            bar_kwargs = dict(color="none", hatch="////", edgecolor=color, linewidth=0)
        else:
            bar_kwargs = dict(color=color, edgecolor="white", linewidth=0)
        for idx in np.flatnonzero(significant):
            left  = freqs_band[idx] - freq_step / 2
            right = freqs_band[idx] + freq_step / 2
            ax_sig.bar((left + right) / 2, height=bar_height, bottom=bottom, width=right - left, **bar_kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.15, 0.15)
    ax.axvline(alpha_range[0], color="0.5", linewidth=0.8, linestyle="--")
    ax.axvline(alpha_range[1], color="0.5", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"Power change")
    ax.set_title(f"Stim On vs Stim Off log power change (channel: {channel}, Laplacian reference)")
    ax.legend(loc="upper right", title="Stim onset")

    ax_sig.set_ylabel("T-stat")
    ax_sig.set_xlim(xlim)
    ax_sig.set_ylim(0, (len(conditions) - 1) * row_spacing + bar_height)
    ax_sig.set_yticks([]) #row * row_spacing + bar_height / 2 for row in range(len(conditions))])
    # ax_sig.set_yticklabels([CONDITION_LABELS.get(c, f"{c}°") for c in conditions], fontsize=7)
    ax_sig.tick_params(axis="y", length=0)
    ax_sig.invert_yaxis()  # row 0 (first in CONDITION_LABELS / legend order) at the top, not the bottom
    ax_sig.axvline(alpha_range[0], color="0.5", linewidth=0.8, linestyle="--")
    ax_sig.axvline(alpha_range[1], color="0.5", linewidth=0.8, linestyle="--")
    ax_sig.set_xlabel("Frequency (Hz)")

    fig.align_ylabels([ax, ax_sig])

    save_pgf(fig, "condition_log_power_change")
    save_png(fig, "condition_log_power_change")


def compute_run_log_ratio(run: RunData, win_s: float = 2.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (freqs, log10(on/off) ratio per Stim On trial) for one already-loaded run."""
    freqs_off, off_psds = compute_welch_psd(run.off_arrays, run.fs, win_s=win_s, n_fft=int(run.fs / 0.1))
    freqs_on,  on_psds  = compute_welch_psd(run.on_arrays,  run.fs, win_s=win_s, n_fft=int(run.fs / 0.1))
    if off_psds.size == 0 or on_psds.size == 0:
        return None

    log_ratio = compute_log_power_change(on_psds, off_psds)
    return freqs_on, log_ratio


def compute_run_tf_log_power_change(
    run: RunData,
    win_s: float = 1.0,
    step_s: float = 1.0,
    smooth_window_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (freqs, times_s, log power change per Stim On trial) for one already-loaded run.

    log power change has shape (n_on_trials, n_times, n_freqs); each Stim On
    trial is baseline-normalized against the run's mean Stim Off Welch PSD.
    """
    _, off_psds = compute_welch_psd(run.off_arrays, run.fs, win_s=win_s)
    if off_psds.size == 0:
        return None
    baseline_psd = off_psds.mean(axis=0)

    freqs = times = None
    trial_maps = []
    for data in run.on_arrays:
        f, t, log_power_change = compute_trial_tf_log_power_change(
            data, baseline_psd, run.fs, win_s=win_s, step_s=step_s, smooth_window_s=smooth_window_s)
        if log_power_change.size == 0:
            continue
        freqs, times = f, t
        trial_maps.append(log_power_change)

    if not trial_maps:
        return None

    n_times_min = min(m.shape[0] for m in trial_maps)
    trial_maps  = np.stack([m[:n_times_min] for m in trial_maps])  # (n_trials, n_times, n_freqs)
    return freqs, times[:n_times_min], trial_maps


def compute_condition_tf_maps(
    condition_degs: tuple[int, int],
    runs_by_condition: dict[int, list[str]],
    loaded_runs: dict[str, RunData],
    win_s: float = 1.0,
    step_s: float = 1.0,
    smooth_window_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pool Stim On trials for two stim_onset_deg conditions across all runs/sessions.

    Returns (freqs, times_s, maps_a, maps_b), each maps_* of shape (n_trials, n_times, n_freqs).
    """
    freqs = times = None
    maps_by_condition = {}
    for condition in condition_degs:
        trial_maps = []
        for run_dir in runs_by_condition.get(condition, []):
            run = loaded_runs.get(run_dir)
            if run is None:
                continue
            result = compute_run_tf_log_power_change(
                run, win_s=win_s, step_s=step_s, smooth_window_s=smooth_window_s)
            if result is None:
                continue
            freqs, times, maps = result
            trial_maps.append(maps)

        if not trial_maps:
            print(f"No usable trials for stim_onset_deg={condition}")
            return None
        maps_by_condition[condition] = trial_maps

    all_maps    = [m for maps in maps_by_condition.values() for m in maps]
    n_times_min = min(m.shape[1] for m in all_maps)

    stacked = {
        condition: np.concatenate([m[:, :n_times_min, :] for m in maps], axis=0)
        for condition, maps in maps_by_condition.items()
    }
    return freqs, times[:n_times_min], stacked[condition_degs[0]], stacked[condition_degs[1]]


def compute_pooled_tf_log_power_change(
    loaded_runs: dict[str, RunData],
    win_s: float = 2.0,
    step_s: float = 1.0,
    smooth_window_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pool Stim On trials across every loaded run, regardless of stim_onset_deg.

    Each trial's map is already a per-run log10(on / off-baseline) power change
    (see compute_run_tf_log_power_change), so pooling here - without splitting
    by condition - gives the overall Stim On vs Stim Off effect in time-frequency
    resolution, independent of stimulation phase.

    Returns (freqs, times_s, log_power_change), log_power_change of shape
    (n_trials, n_times, n_freqs).
    """
    freqs = times = None
    trial_maps = []
    for run in loaded_runs.values():
        result = compute_run_tf_log_power_change(run, win_s=win_s, step_s=step_s, smooth_window_s=smooth_window_s)
        if result is None:
            continue
        freqs, times, maps = result
        trial_maps.append(maps)

    if not trial_maps:
        return None

    n_times_min = min(m.shape[1] for m in trial_maps)
    log_power_change = np.concatenate([m[:, :n_times_min, :] for m in trial_maps], axis=0)
    return freqs, times[:n_times_min], log_power_change


def compute_tf_ttest(maps_a: np.ndarray, maps_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent two-sample t-test between two trial groups, at each (time, freq) bin.

    Returns (t_stat, p_value), each of shape (n_times, n_freqs).
    """
    return ttest_ind(maps_a, maps_b, axis=0)


def plot_tf_ttest(
    freqs: np.ndarray,
    times: np.ndarray,
    t_stat: np.ndarray,
    title: str,
    n_a: int,
    n_b: int,
    ylim: tuple = (4, 17),
    clim: tuple = (-5, 5),
) -> None:
    """Plot a time x frequency t-statistic map (fig_2.m panel B style)."""
    fig  = plt.figure(figsize=FIGSIZE)
    mesh = plt.pcolormesh(times, freqs, t_stat.T, shading="nearest",
                           cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    plt.colorbar(mesh, label="t-statistic")

    plt.axhline(7.5,  color="0", linestyle="--", linewidth=0.8)
    plt.axhline(12.5, color="0", linestyle="--", linewidth=0.8)

    plt.ylim(ylim)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"{title} (Stim On, n={n_a} vs n={n_b})")

    save_pgf(fig, f"tf_ttest_{title.lower().replace(' ', '_').replace('/', '_')}")


def plot_tf_power_change(
    freqs: np.ndarray,
    times: np.ndarray,
    log_power_change: np.ndarray,
    ylim: tuple = (4, 17),
    clim: tuple = (-0.6, 0.6),
) -> None:
    """Plot the mean time x frequency log10(on/off) power change, pooled across all conditions."""
    fig  = plt.figure(figsize=FIGSIZE)
    mean = log_power_change.mean(axis=0)
    mesh = plt.pcolormesh(times, freqs, mean.T, shading="nearest",
                           cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    plt.colorbar(mesh, label="Power change (log10 on/off)")

    plt.axhline(7.5,  color="0", linestyle="--", linewidth=0.8)
    plt.axhline(12.5, color="0", linestyle="--", linewidth=0.8)

    plt.ylim(ylim)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Stim On vs Stim Off power change, all conditions pooled (n={log_power_change.shape[0]})")

    save_pgf(fig, "tf_power_change_pooled")


def analyse_spectrum(results_dir: str, channel: str = "Fz", win_s: float = 2.0) -> None:
    """Ad-hoc single-run version of the log power change spectrum plot."""
    run = load_run(results_dir, channel=channel)
    if run is None:
        return
    result = compute_run_log_ratio(run, win_s=win_s)
    if result is None:
        return
    freqs, log_ratio = result
    plot_log_power_change(freqs, log_ratio, channel)
    plt.show()


def analyse_all(
    base_dir: str = "results",
    stim_onset_degs: tuple[int, ...] = STIM_ONSET_DEGS,
    comparisons: tuple[tuple[int, int], ...] = TF_COMPARISONS,
    channel: str = "Fz",
    win_s: float = 1.0,
    step_s: float = 1.0,
    smooth_window_s: float = 10.0,
) -> None:
    """Produce both fig_2.m-style plots without loading/preprocessing any run twice.

    1. Discover every run once and load (EDF read, CSD reference, IAF-based
       cleaning + lowpass filter, epoching) each of them exactly once.
    2. Plot the log power change spectrum per stim_onset_deg condition (panel C).
    3. Plot the opposite-phase time-frequency t-maps (panel B), reusing the same
       loaded runs.
    """
    runs_by_condition = discover_condition_runs(base_dir, stim_onset_degs)

    all_run_dirs = sorted({run_dir for run_dirs in runs_by_condition.values() for run_dir in run_dirs})
    loaded_runs  = {run_dir: run for run_dir in all_run_dirs
                    if (run := load_run(run_dir, channel=channel)) is not None}

    # --- Plot 1: log power change spectrum per condition (fig_2.m panel C) ---
    freqs = None
    log_ratio_by_condition = {}
    for condition in stim_onset_degs:
        run_dirs = runs_by_condition[condition]
        if not run_dirs:
            print(f"No runs found for stim_onset_deg={condition}")
            continue

        n_sessions = len({Path(run_dir).parent for run_dir in run_dirs})
        print(f"stim_onset_deg={condition}: {len(run_dirs)} run(s) across {n_sessions} session folder(s)")
        condition_ratios = []
        for run_dir in run_dirs:
            run = loaded_runs.get(run_dir)
            if run is None:
                continue
            result = compute_run_log_ratio(run, win_s=win_s)
            if result is None:
                continue
            freqs, log_ratio = result
            condition_ratios.append(log_ratio)

        if condition_ratios:
            log_ratio_by_condition[condition] = np.concatenate(condition_ratios, axis=0)

    if freqs is not None and log_ratio_by_condition:
        plot_condition_log_power_change(freqs, log_ratio_by_condition, channel)
    else:
        print("No usable runs found for the spectrum plot.")

    # --- Plot 2: opposite-phase time-frequency t-maps (fig_2.m panel B) ---
    for deg_a, deg_b in comparisons:
        title  = f"{CONDITION_LABELS.get(deg_a, deg_a)} vs {CONDITION_LABELS.get(deg_b, deg_b)}"
        result = compute_condition_tf_maps(
            (deg_a, deg_b), runs_by_condition, loaded_runs,
            win_s=win_s, step_s=step_s, smooth_window_s=smooth_window_s)
        if result is None:
            print(f"Skipping {title}: not enough data.")
            continue

        freqs_tf, times_tf, maps_a, maps_b = result
        t_stat, _ = compute_tf_ttest(maps_a, maps_b)
        plot_tf_ttest(freqs_tf, times_tf, t_stat, title, n_a=maps_a.shape[0], n_b=maps_b.shape[0])

    # --- Plot 3: pooled Stim On vs Stim Off power change, all conditions combined ---
    result = compute_pooled_tf_log_power_change(
        loaded_runs, win_s=win_s, step_s=step_s, smooth_window_s=smooth_window_s)
    if result is None:
        print("Skipping pooled Stim On vs Stim Off plot: not enough data.")
    else:
        freqs_pooled, times_pooled, log_power_change = result
        plot_tf_power_change(freqs_pooled, times_pooled, log_power_change)

    plt.show()


if __name__ == "__main__":
    analyse_all()
