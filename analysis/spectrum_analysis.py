"""
Draft analysis: log power change (Stim On vs Stim Off) per frequency bin,
grouped by stim_onset_deg condition.
    python -m analysis.spectrum_analysis   (from CLAS/)

Discovers run directories under results/CLAS*/*/, reads each run's graph
yaml to find its stim_onset_deg condition, loads the EDF file written by
analysis/main.py (via write_edf), splits the recording into "stim_on" /
"stim_off" segments using the annotations added by load_stim_annotations,
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
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.plot import FIGSIZE, save_pgf

CONDITIONS      = ("stim_on", "stim_off")
STIM_ONSET_DEGS = (0, 60, 150, 180, 240, 330)

CONDITION_LABELS = {
    0:   "P1-Peak",
    60:  "Post-Peak",
    150: "Pre-Trough",
    180: "P1-Trough",
    240: "Post-Trough",
    330: "Pre-Peak",
}
CONDITION_COLORS = {
    0:   "lightgreen",
    60:  "red",
    150: "green",
    180: "lightblue",
    240: "orange",
    330: "blue",
}

# Opposite-phase comparisons for the time-frequency t-maps (fig_2.m panel B):
# Pre-Peak vs Pre-Trough, and Post-Peak vs Post-Trough.
TF_COMPARISONS = ((330, 150), (60, 240))


def find_edf_path(results_dir: str) -> str:
    """Resolve the EDF file for a results directory (mirrors analysis/main.py)."""
    results_path = Path(results_dir)
    edf_stem = results_path.readlink().stem if results_path.is_symlink() else results_path.stem
    edf_path = results_path / f"{edf_stem}.edf"
    if edf_path.exists():
        return str(edf_path)

    matches = glob.glob(str(results_path / "*.edf"))
    if not matches:
        raise FileNotFoundError(f"No EDF file found in {results_dir}")
    return matches[0]


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
    """Match a channel by its base name, ignoring the type prefix EDF export adds."""
    if name in raw.ch_names:
        return name
    matches = [ch for ch in raw.ch_names if ch.split(" ", 1)[-1] == name]
    if not matches:
        raise ValueError(f"Channel '{name}' not found in {raw.ch_names}")
    return matches[0]


def add_laplacian_reference(
    raw: "mne.io.BaseRaw",
    channel: int = 3,
    old_ref: int = 10,
    neighbors: tuple[int, ...] = (1, 2, 29, 27, 26, 4),
    new_name: str = "Laplacian",
) -> "mne.io.BaseRaw":
    """Re-reference an EEG_N channel from its old online (hardware) reference to a
    Laplacian (local average) reference, and add the result as a new channel.

    `channel` was referenced online, by the amplifier hardware, against `old_ref`
    during acquisition. That reference is removed first (channel - old_ref), then
    a new Laplacian reference is applied by subtracting the mean of `neighbors`.
    All indices refer to the EEG_N channel numbering used in the EDF export.
    """
    def _pick(idx: int) -> np.ndarray:
        return raw.get_data(picks=resolve_channel(raw, f"EEG_{idx}"))[0]

    unreferenced = _pick(channel) + _pick(old_ref)
    laplacian    = unreferenced - np.mean([_pick(n) for n in neighbors], axis=0)

    info    = mne.create_info([new_name], sfreq=raw.info["sfreq"], ch_types="eeg", verbose=False)
    new_raw = mne.io.RawArray(laplacian[np.newaxis, :], info, verbose=False)
    new_raw.set_meas_date(raw.info["meas_date"])
    raw.add_channels([new_raw], force_update_info=True)
    return raw


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
    iaf_channel: str = "IAF_est_Hz",
    lowpass_hz: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop samples where the online IAF estimate is zero, then lowpass filter.

    Concatenating across the dropped stretches introduces edge discontinuities;
    the lowpass filter, applied after concatenation and before epoching, smooths
    those edges out and prevents them from leaking into high frequencies.

    Returns (cleaned_signal, retained_idx), where retained_idx[i] is the sample
    index in the original (uncleaned) recording that cleaned_signal[i] came from
    - used to translate stim_on/stim_off sample ranges onto the cleaned timeline.
    """
    fs     = raw.info["sfreq"]
    signal = raw.get_data(picks=resolve_channel(raw, channel))[0] * 1e6  # V -> uV
    iaf    = raw.get_data(picks=resolve_channel(raw, iaf_channel))[0]

    retained_idx   = np.flatnonzero(iaf != 0)
    cleaned_signal = signal[retained_idx]
    cleaned_signal = mne.filter.filter_data(cleaned_signal, fs, l_freq=None, h_freq=lowpass_hz, verbose=False)

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
    channel: str = "Laplacian",
    lowpass_hz: float = 30.0,
) -> RunData | None:
    """Load, re-reference, clean, and epoch one run's EDF into Stim On/Off sample arrays.

    This is the expensive, shared preprocessing step (EDF read, Laplacian
    reference, IAF-based cleaning + lowpass filter) needed by both the
    per-condition spectrum plot and the time-frequency t-maps -- run it once
    per run directory and reuse the result for both.
    """
    edf_path = find_edf_path(results_dir)
    raw      = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    add_laplacian_reference(raw, channel=3, old_ref=10, neighbors=(1, 2, 29, 27, 26, 4))
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


def plot_log_power_change(
    freqs: np.ndarray,
    log_ratio: np.ndarray,
    channel: str,
    xlim: tuple = (6, 13),
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
    plt.legend(frameon=True)

    save_pgf(fig, "stim_on_off_log_power_change")


def plot_condition_log_power_change(
    freqs: np.ndarray,
    log_ratio_by_condition: dict[int, np.ndarray],
    channel: str,
    xlim: tuple = (0, 30),
) -> None:
    """Plot mean +/- SEM log10(on / off) power ratio per stim_onset_deg condition."""
    fig = plt.figure(figsize=FIGSIZE)

    plt.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    for condition, log_ratio in log_ratio_by_condition.items():
        color = CONDITION_COLORS.get(condition, "0.3")
        label = CONDITION_LABELS.get(condition, f"{condition}°")
        mean  = log_ratio.mean(axis=0)
        sem   = log_ratio.std(axis=0) / np.sqrt(log_ratio.shape[0])
        plt.plot(freqs, mean, color=color, label=f"{label} (n={log_ratio.shape[0]})")
        plt.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.2)

    plt.xlim(xlim)
    plt.axvline(6, color="0.5", linewidth=0.8, linestyle="--")
    plt.axvline(14, color="0.5", linewidth=0.8, linestyle="--")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(f"Power change ({channel}, log10 on/off)")
    plt.title("Stim On vs Stim Off log power change by stimulation phase")
    plt.legend(frameon=True, title="Stim onset")

    save_pgf(fig, "condition_log_power_change")


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


def analyse_spectrum(results_dir: str, channel: str = "Laplacian", win_s: float = 2.0) -> None:
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
    channel: str = "Laplacian",
    win_s: float = 1.0,
    step_s: float = 1.0,
    smooth_window_s: float = 10.0,
) -> None:
    """Produce both fig_2.m-style plots without loading/preprocessing any run twice.

    1. Discover every run once and load (EDF read, Laplacian reference, IAF-based
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
