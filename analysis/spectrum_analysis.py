"""
Log power change (Stim On vs Stim Off), both per frequency bin and
time-resolved (time x frequency), grouped by stim_onset_deg condition.
    python -m analysis.spectrum_analysis   (from CLAS/)

Discovers recording directories under results/CLAS_*/*/, reads each
recording's graph yaml to find its stim_onset_deg condition, loads
raw_signals.edf (written by analysis/edf_io.py:write_raw_signals_edf, via
analysis/main.py) and runtime_metadata.h5 (for an online-f0 validity sanity
check), splits the recording into "stim_on" / "stim_off" segments using the
annotations added by load_stim_annotations, and computes a Welch PSD per
segment on one channel.

Every per-recording quantity is averaged within its own recording (Stim
On/Off segments, or Stim On trials) before being pooled across recordings --
see compute_recording_log_ratio -- so a recording with more segments/trials
doesn't dominate the pooled result. Produces:

- a log10(mean(on)/mean(off)) power-ratio spectrum, mean +/- std across
  recordings, per stim_onset_deg condition, with a per-frequency
  significance track (one-sample t-test against 0 across recordings);
- the same ratio time-resolved, mean across recordings, as one grid figure
  with one subplot per stim_onset_deg condition.
"""

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

from analysis.loader import find_session_run_dirs
from analysis.plot import FIGSIZE, FIG_WIDTH, FIG_HEIGHT, save_pgf, save_pdf, CHANNEL_NAMES, EOG_CHANNEL_NAMES
from analysis.runtime_meta import load_runtime_metadata

CONDITIONS      = ("stim_on", "stim_off")
STIM_ONSET_DEGS = (0, 60, 150, 180, 240, 330)

CONDITION_LABELS = {
    330: "Pre-Peak",
    180: r"$\mathrm{P}_1$-Trough",
    60:  "Post-Peak",
    150: "Pre-Trough",
    0:   r"$\mathrm{P}_1$-Peak",
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

# The two P1 conditions (peak/trough itself) are dashed to stand out from the
# four solid phase-offset conditions
CONDITION_LINESTYLES = {0: "--", 180: "--"}

# Opposite-phase comparisons for the time-frequency t-maps:
# Pre-Peak vs Pre-Trough, and Post-Peak vs Post-Trough.
TF_COMPARISONS = ((330, 150), (60, 240))

# Extended alpha band the per-condition significance test is restricted to.
EXTENDED_ALPHA_RANGE = (6, 14)

# Zero-padded Welch frequency resolution (Hz), shared so plots 1 and 3 report the same frequency axis
SPECTRAL_RESOLUTION_HZ = 0.1


def discover_condition_recordings(
    base_dir: str = "results",
    stim_onset_degs: tuple[int, ...] = STIM_ONSET_DEGS,
) -> dict[int, list[str]]:
    """Group recording directories by their stim_onset_deg condition.

    Pools across every top-level CLAS_*-prefixed session folder found under
    base_dir (e.g. multiple subjects/sessions), not just a single one.
    """
    recordings_by_condition = {deg: [] for deg in stim_onset_degs}

    recording_dirs = find_session_run_dirs(base_dir)
    session_dirs = sorted({os.path.dirname(recording_dir) for recording_dir in recording_dirs})
    if len(session_dirs) > 1:
        print(f"Pooling across {len(session_dirs)} CLAS_* session folders: "
              f"{[os.path.basename(d) for d in session_dirs]}")

    for recording_dir in recording_dirs:
        with open(os.path.join(recording_dir, "TurboLinkCLAS.yaml")) as f:
            graph_config = yaml.safe_load(f)
        stim_onset_deg = (
            graph_config.get("graph", {})
            .get("processors", {})
            .get("StimControl", {})
            .get("options", {})
            .get("stim_onset_deg")
        )

        if stim_onset_deg in recordings_by_condition:
            recordings_by_condition[stim_onset_deg].append(recording_dir)
        elif stim_onset_deg is not None:
            print(f"Skipping {recording_dir}: stim_onset_deg={stim_onset_deg} not in {stim_onset_degs}")

    return recordings_by_condition


def apply_csd_reference(raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
    """Replace the online (hardware) reference with a surface Laplacian /
    current source density (CSD) reference, computed across the channels with
    a known 10-20 position (see CHANNEL_NAMES).

    CSD is reference-free, so the recorded signal is used directly. Channels
    without a known position are dropped (compute_current_source_density
    requires a 10-20 site for every "eeg"-typed channel); E1/E2 are
    relabelled "eog" instead, to exclude them from the CSD computation.

    Channels are identified by name rather than raw.get_channel_types():
    the EDF round-trip loses the original channel types, so every channel
    comes back typed "eeg".
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


@dataclass
class RecordingData:
    fs: float
    on_arrays: list[np.ndarray]
    off_arrays: list[np.ndarray]
    f0_invalid_pct: float | None = None


def compute_f0_invalid_pct(results_dir: str, segments: dict[str, list[tuple[int, int]]]) -> float | None:
    """Percentage of stim_on samples with an invalid (NaN) online f0 estimate
    (see FrequencyEstimation.cpp), from runtime_metadata.h5's "f0" field.

    Returns None if runtime_metadata.h5 or its f0 field isn't available for
    this recording, or if there are no stim_on samples to check.
    """
    meta_h5_path = os.path.join(results_dir, "runtime_metadata.h5")
    if not os.path.exists(meta_h5_path):
        return None
    f0 = load_runtime_metadata(meta_h5_path)["online"].get("f0")
    if f0 is None:
        return None

    n_invalid = n_total = 0
    for start_idx, end_idx in segments["stim_on"]:
        end_idx = min(end_idx, len(f0))
        if end_idx <= start_idx:
            continue
        n_invalid += int(np.isnan(f0[start_idx:end_idx]).sum())
        n_total   += end_idx - start_idx

    return 100 * n_invalid / n_total if n_total else None


def load_recording(
    results_dir: str,
    channel: str = "Fz",
    lowpass_hz: float = 30.0,
    use_csd: bool = True,
) -> RecordingData | None:
    """Load, re-reference, filter, and epoch one recording's EDF into Stim On/Off sample arrays.

    `use_csd` toggles the surface Laplacian / CSD re-referencing step (see
    apply_csd_reference); when False, `channel`'s signal is used as recorded
    (still under whatever online/hardware reference the EDF was written
    with), since apply_csd_reference is the only re-referencing done here.
    """
    # All-channel EEG data lives only in raw_signals.edf -- analysis.edf only carries the single selected channel.
    edf_path = Path(results_dir) / "raw_signals.edf"
    if not edf_path.exists():
        raise FileNotFoundError(f"No raw_signals.edf found in {results_dir}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    if use_csd:
        apply_csd_reference(raw)
    fs = raw.info["sfreq"]

    # raw_signals.edf/analysis.edf are written with add_ch_type=False, so labels carry no type prefix to strip.
    if channel not in raw.ch_names:
        raise ValueError(f"Channel '{channel}' not found in {raw.ch_names}")
    signal = raw.get_data(picks=channel)[0] * 1e6  # V -> uV

    # signal = mne.filter.filter_data(signal, fs, l_freq=None, h_freq=lowpass_hz, verbose=False)

    segments = get_condition_segments(raw)

    f0_invalid_pct = compute_f0_invalid_pct(results_dir, segments)
    if f0_invalid_pct is not None:
        print(f"  {results_dir}: {f0_invalid_pct:.1f}% of stim-on samples had invalid (NaN) f0")

    on_arrays  = [signal[s:e] for s, e in segments["stim_on"]]
    off_arrays = [signal[s:e] for s, e in segments["stim_off"]]
    print(f"  {results_dir}: stim_on: {len(on_arrays)} segment(s), stim_off: {len(off_arrays)} segment(s)")

    if not on_arrays or not off_arrays:
        return None
    return RecordingData(fs=fs, on_arrays=on_arrays, off_arrays=off_arrays, f0_invalid_pct=f0_invalid_pct)


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
    n_fft: int | None = None,
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
        # Equals periodogram, n_segments=1, because nperseg = len(data)
        f, pxx = welch(data[start:start + nperseg], fs=fs, nperseg=nperseg, window="hamming", nfft=n_fft)
        freqs = f
        psds.append(pxx)
        times.append((start + nperseg / 2) / fs)
        start += step

    return freqs, np.array(times), np.array(psds) if psds else np.empty((0, 0))


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
    """Plot one recording's log10(mean(on)/mean(off)) power ratio per frequency bin."""
    fig  = plt.figure(figsize=FIGSIZE)

    plt.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    plt.plot(freqs, log_ratio, color="tab:purple", label="log10(on/off)", rasterized=True)

    plt.xlim(xlim)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(f"Power change ({channel}, log10 on/off)")
    plt.title("Stim On vs Stim Off log power change")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)
    plt.legend(loc="upper left")

    save_pdf(fig, "stim_on_off_log_power_change")
    # save_pgf(fig, "stim_on_off_log_power_change")


def plot_condition_log_power_change(
    freqs: np.ndarray,
    log_ratio_by_condition: dict[int, np.ndarray],
    xlim: tuple = (4, 17),
    alpha_range: tuple = EXTENDED_ALPHA_RANGE,
) -> None:
    """Plot mean +/- std log10(on / off) power ratio per stim_onset_deg condition,
    across recordings (each recording contributes one log10(mean(on)/mean(off))
    spectrum -- see compute_recording_log_ratio), with a significance track below
    marking the frequencies (within alpha_range) where a condition's power
    change differs significantly from zero across recordings (one-sample
    t-test, p < 0.05, uncorrected)."""
    conditions  = [c for c in CONDITION_LABELS if c in log_ratio_by_condition]
    row_spacing = 0.1  # was 1.0 -- distance between row centers, packs the bars together
    bar_height  = 0.09  # was 0.8 -- leaves a thin gap between rows at this spacing
    height_ratios = [8, 1]
    fig, (ax, ax_sig) = plt.subplots(
        2, 1, figsize=FIGSIZE, sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0},
    )

    ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    freq_step = np.median(np.diff(freqs))
    for row, condition in enumerate(conditions):
        log_ratio = log_ratio_by_condition[condition]
        color     = CONDITION_COLORS.get(condition, "0.3")
        linestyle = CONDITION_LINESTYLES.get(condition, "-")
        label     = CONDITION_LABELS.get(condition, f"{condition} deg")
        mean = log_ratio.mean(axis=0)
        # std  = log_ratio.std(axis=0)
        ax.plot(freqs, mean, color=color, linestyle=linestyle, label=f"{label} ({condition} deg)", rasterized=True)
        # ax.fill_between(freqs, mean - std, mean + std, color=color, alpha=0.2)

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
            ax_sig.bar((left + right) / 2, height=bar_height, bottom=bottom, width=right - left,
                       rasterized=True, **bar_kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.15, 0.15)
    ax.axvline(alpha_range[0], color="0.5", linewidth=0.8, linestyle="--")
    ax.axvline(alpha_range[1], color="0.5", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"Power change (log10 on/off)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

    ax_sig.set_xlim(xlim)
    ax_sig.set_ylim(0, (len(conditions) - 1) * row_spacing + bar_height)
    ax_sig.set_yticks([]) #row * row_spacing + bar_height / 2 for row in range(len(conditions))])
    # ax_sig.set_yticklabels([CONDITION_LABELS.get(c, f"{c} deg") for c in conditions], fontsize=7)
    ax_sig.tick_params(axis="y", length=0)
    ax_sig.invert_yaxis()  # row 0 (first in CONDITION_LABELS / legend order) at the top, not the bottom
    ax_sig.axvline(alpha_range[0], color="0.5", linewidth=0.8, linestyle="--")
    ax_sig.axvline(alpha_range[1], color="0.5", linewidth=0.8, linestyle="--")
    ax_sig.set_xlabel("Frequency [Hz]")

    fig.align_ylabels([ax, ax_sig])

    # Reserve room below the axes (xlabel included) for the legend, tuned so the
    # xlabel-to-legend gap matches plot_tf_power_change_grid's xlabel-to-colorbar gap.
    fig.subplots_adjust(bottom=0.139)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)

    # save_pgf(fig, "condition_log_power_change")
    save_pdf(fig, "condition_log_power_change")


def compute_recording_log_ratio(
    recording: RecordingData,
    win_s: float = 1.0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (freqs, log10(mean(on PSD) / mean(off PSD))) for one already-loaded
    recording: a single per-recording spectrum, averaging over that recording's
    Stim On/Off windows before taking the ratio.

    """
    n_fft = int(recording.fs / SPECTRAL_RESOLUTION_HZ)
    freqs, off_psds = compute_welch_psd(recording.off_arrays, recording.fs, win_s=win_s, n_fft=n_fft)
    if off_psds.size == 0:
        return None

    _, on_psds = compute_welch_psd(recording.on_arrays, recording.fs, win_s=win_s, n_fft=n_fft)
    if on_psds.size == 0:
        return None

    log_ratio = np.log10(on_psds.mean(axis=0) / off_psds.mean(axis=0))
    return freqs, log_ratio


def compute_recording_tf_log_power_change(
    recording: RecordingData,
    win_s: float = 1.0,
    step_s: float = 1.0,
    smooth_window_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (freqs, times_s, log power change) for one already-loaded recording:
    a single time-resolved map, shape (n_times, n_freqs).

    Averages Stim On PSDs across trials, then takes the log ratio against the
    mean Stim Off PSD, matching compute_recording_log_ratio's "ratio of means"
    ordering (mean(log(x)) != log(mean(x))).
    """
    n_fft = int(recording.fs / SPECTRAL_RESOLUTION_HZ)
    _, off_psds = compute_welch_psd(recording.off_arrays, recording.fs, win_s=win_s, n_fft=n_fft)
    if off_psds.size == 0:
        return None
    baseline_psd = off_psds.mean(axis=0)

    freqs = times = None
    trial_psds = []
    for data in recording.on_arrays:
        freqs, times, psd = compute_welch_spectrogram(data, recording.fs, win_s=win_s, step_s=step_s, n_fft=n_fft)
        if psd.size == 0:
            continue
        trial_psds.append(psd)

    if not trial_psds:
        return None

    # Truncate all trials to the shortest length, because they may differ by milliseconds (stim protocol jitter)
    n_times_min = min(p.shape[0] for p in trial_psds)
    mean_psd = np.stack([p[:n_times_min] for p in trial_psds]).mean(axis=0)  # (n_times, n_freqs)

    log_power_change = np.log10(mean_psd / baseline_psd)

    # Centered moving average over time (movmean-style: `half` points on each side, edges shrink).
    smooth_window = int(round(smooth_window_s / step_s))
    half = smooth_window // 2
    smoothed = np.empty_like(log_power_change)
    for i in range(len(log_power_change)):
        lo, hi = max(0, i - half), min(len(log_power_change), i + half + 1)
        smoothed[i] = log_power_change[lo:hi].mean(axis=0)
    log_power_change = smoothed

    return freqs, times[:n_times_min], log_power_change


def compute_condition_tf_maps(
    condition_degs: tuple[int, int],
    recordings_by_condition: dict[int, list[str]],
    loaded_recordings: dict[str, RecordingData],
    win_s: float = 1.0,
    step_s: float = 1.0,
    smooth_window_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-recording time-frequency maps for two stim_onset_deg conditions,
    pooled across all recordings/sessions -- one map per recording (see
    compute_recording_tf_log_power_change), not one per trial, so recordings
    with more trials don't dominate the pooled t-test.

    Returns (freqs, times_s, maps_a, maps_b), each maps_* of shape (n_recordings, n_times, n_freqs).
    """
    freqs = times = None
    maps_by_condition = {}
    for condition in condition_degs:
        recording_maps = []
        for recording_dir in recordings_by_condition.get(condition, []):
            recording = loaded_recordings.get(recording_dir)
            if recording is None:
                continue
            result = compute_recording_tf_log_power_change(
                recording, win_s=win_s, step_s=step_s, smooth_window_s=smooth_window_s)
            if result is None:
                continue
            freqs, times, mean_map = result
            recording_maps.append(mean_map)

        if not recording_maps:
            print(f"No usable recordings for stim_onset_deg={condition}")
            return None
        maps_by_condition[condition] = recording_maps

    all_maps    = [m for maps in maps_by_condition.values() for m in maps]
    n_times_min = min(m.shape[0] for m in all_maps)

    stacked = {
        condition: np.stack([m[:n_times_min, :] for m in maps], axis=0)
        for condition, maps in maps_by_condition.items()
    }
    return freqs, times[:n_times_min], stacked[condition_degs[0]], stacked[condition_degs[1]]


def compute_pooled_tf_log_power_change(
    loaded_recordings: dict[str, RecordingData],
    recordings_by_condition: dict[int, list[str]] | None = None,
    conditions: tuple[int, ...] | None = None,
    win_s: float = 1.0,
    step_s: float = 1.0,
    smooth_window_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pool per-recording time-frequency maps across loaded recordings.

    Pools every loaded recording by default; pass `conditions` (a subset of
    stim_onset_deg values) with `recordings_by_condition` (from
    discover_condition_recordings) to restrict to those conditions.

    Each recording contributes one map (see
    compute_recording_tf_log_power_change), so a recording with many trials
    doesn't dominate the pooled mean.

    Returns:
        (freqs, times_s, log_power_change), the latter shape
        (n_recordings, n_times, n_freqs).
    """
    if conditions is None:
        recordings = loaded_recordings
    else:
        if recordings_by_condition is None:
            raise ValueError("recordings_by_condition is required when conditions is given")
        wanted_recording_dirs = {recording_dir for condition in conditions
                                  for recording_dir in recordings_by_condition.get(condition, [])}
        recordings = {recording_dir: recording for recording_dir, recording in loaded_recordings.items()
                      if recording_dir in wanted_recording_dirs}

    freqs = times = None
    recording_maps = []
    for recording in recordings.values():
        result = compute_recording_tf_log_power_change(
            recording, win_s=win_s, step_s=step_s, smooth_window_s=smooth_window_s)
        if result is None:
            continue
        freqs, times, mean_map = result
        recording_maps.append(mean_map)

    if not recording_maps:
        return None

    n_times_min = min(m.shape[0] for m in recording_maps)
    log_power_change = np.stack([m[:n_times_min, :] for m in recording_maps], axis=0)
    return freqs, times[:n_times_min], log_power_change


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
    """Plot a time x frequency t-statistic map."""
    fig  = plt.figure(figsize=FIGSIZE)
    mesh = plt.pcolormesh(times, freqs, t_stat.T, shading="nearest",
                           cmap="RdBu_r", vmin=clim[0], vmax=clim[1], rasterized=True)
    plt.colorbar(mesh, label="t-statistic")

    plt.axhline(EXTENDED_ALPHA_RANGE[0], color="0", linestyle="--", linewidth=0.8)
    plt.axhline(EXTENDED_ALPHA_RANGE[1], color="0", linestyle="--", linewidth=0.8)

    plt.ylim(ylim)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(f"{title} (Stim On, n_recordings={n_a} vs {n_b})")

    save_pdf(fig, f"tf_ttest_{title.lower().replace(' ', '_').replace('/', '_')}")


def plot_tf_power_change_grid(
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ncols: int = 3,
    ylim: tuple = (4, 17),
    clim: tuple | None = None,
    clim_percentile: float = 99.0,
) -> None:
    """Grid of mean time x frequency log10(on/off) power change maps, one
    subplot per condition, sharing x/y axes and a single colorbar.

    Args:
        results: Maps a subplot title to that condition's
            (freqs, times, log_power_change) triple, as returned by
            compute_pooled_tf_log_power_change.
        clim: Fixes the symmetric colorbar range; None derives it from
            `clim_percentile` of |mean power change| within `ylim`, pooled
            across every condition being plotted.
    """
    n     = len(results)
    nrows = -(-n // ncols)  # ceil

    means = {title: log_power_change.mean(axis=0) for title, (_, _, log_power_change) in results.items()}
    if clim is None:
        freqs0    = next(iter(results.values()))[0]
        band      = (freqs0 >= ylim[0]) & (freqs0 <= ylim[1])
        bound     = np.percentile(np.abs(np.concatenate([m[:, band] for m in means.values()])), clim_percentile)
        clim      = (-bound, bound)

    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.5),
                              sharex=True, sharey=True, squeeze=False)

    mesh = None
    for ax, (title, (freqs, times, log_power_change)) in zip(axes.flat, results.items()):
        mean = means[title]
        mesh = ax.pcolormesh(times, freqs, mean.T, shading="nearest",
                              cmap="RdBu_r", vmin=clim[0], vmax=clim[1], rasterized=True)
        ax.axhline(EXTENDED_ALPHA_RANGE[0], color="0", linestyle="--", linewidth=0.8)
        ax.axhline(EXTENDED_ALPHA_RANGE[1], color="0", linestyle="--", linewidth=0.8)
        ax.set_title(title)

    for ax in axes.flat[n:]:
        ax.set_axis_off()

    axes.flat[0].set_ylim(ylim)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency [Hz]")

    fig.colorbar(mesh, ax=axes, label="Power change (log10 on/off)", orientation="horizontal",
                 location="bottom", shrink=0.8, pad=0.12)

    save_pdf(fig, "tf_power_change_pooled")


def analyse_spectrum(results_dir: str, channel: str = "Fz", win_s: float = 1.0, use_csd: bool = True) -> None:
    """Ad-hoc single-recording version of the log power change spectrum plot."""
    recording = load_recording(results_dir, channel=channel, use_csd=use_csd)
    if recording is None:
        return
    result = compute_recording_log_ratio(recording, win_s=win_s)
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
    use_csd: bool = True,
) -> None:
    """Load every recording once (EDF read, CSD reference, lowpass filter,
    epoching) and produce all three plots from it:

    1. Log power change spectrum per stim_onset_deg condition.
    2. Opposite-phase time-frequency t-maps.
    3. Pooled Stim On vs Stim Off power change across `stim_onset_degs`
       (pass a subset, e.g. (0, 180), to restrict all plots to it).

    `win_s`/`step_s` set the shared Welch window/hop for every plot: the
    spectrum is the time-resolved map averaged over time and Stim On segments
    before the ratio (see compute_recording_log_ratio), so all plots share
    one frequency axis via SPECTRAL_RESOLUTION_HZ.

    `use_csd` toggles CSD re-referencing (see load_recording/apply_csd_reference).
    """
    recordings_by_condition = discover_condition_recordings(base_dir, stim_onset_degs)

    all_recording_dirs = sorted({recording_dir for recording_dirs in recordings_by_condition.values()
                                  for recording_dir in recording_dirs})
    loaded_recordings   = {recording_dir: recording for recording_dir in all_recording_dirs
                            if (recording := load_recording(recording_dir, channel=channel, use_csd=use_csd)) is not None}

    f0_invalid_pcts = [recording.f0_invalid_pct for recording in loaded_recordings.values()
                        if recording.f0_invalid_pct is not None]
    if f0_invalid_pcts:
        print(f"Invalid f0 in stim-on periods across {len(f0_invalid_pcts)} recording(s): "
              f"mean={np.mean(f0_invalid_pcts):.1f}%, std={np.std(f0_invalid_pcts):.1f}%")

    # --- Plot 1: log power change spectrum per condition ---
    freqs = None
    log_ratio_by_condition = {}
    for condition in stim_onset_degs:
        recording_dirs = recordings_by_condition[condition]
        if not recording_dirs:
            print(f"No recordings found for stim_onset_deg={condition}")
            continue

        n_sessions = len({Path(recording_dir).parent for recording_dir in recording_dirs})
        print(f"stim_onset_deg={condition}: {len(recording_dirs)} recording(s) across {n_sessions} session folder(s)")
        condition_ratios = []  # one log10(mean(on)/mean(off)) spectrum per recording
        for recording_dir in recording_dirs:
            recording = loaded_recordings.get(recording_dir)
            if recording is None:
                continue
            result = compute_recording_log_ratio(recording, win_s=win_s)
            if result is None:
                continue
            freqs, log_ratio = result
            condition_ratios.append(log_ratio)

        if condition_ratios:
            log_ratio_by_condition[condition] = np.stack(condition_ratios, axis=0)

    if freqs is not None and log_ratio_by_condition:
        plot_condition_log_power_change(freqs, log_ratio_by_condition)
    else:
        print("No usable recordings found for the spectrum plot.")

    # --- Plot 2: opposite-phase time-frequency t-maps ---
    for deg_a, deg_b in comparisons:
        title  = f"{CONDITION_LABELS.get(deg_a, deg_a)} vs {CONDITION_LABELS.get(deg_b, deg_b)}"
        result = compute_condition_tf_maps(
            (deg_a, deg_b), recordings_by_condition, loaded_recordings,
            win_s=win_s, step_s=step_s, smooth_window_s=smooth_window_s)
        if result is None:
            print(f"Skipping {title}: not enough data.")
            continue

        freqs_tf, times_tf, maps_a, maps_b = result
        # Independent two-sample t-test between the two conditions' per-recording maps, at each (time, freq) bin.
        t_stat, _ = ttest_ind(maps_a, maps_b, axis=0)
        plot_tf_ttest(freqs_tf, times_tf, t_stat, title, n_a=maps_a.shape[0], n_b=maps_b.shape[0])

    # --- Plot 3: pooled Stim On vs Stim Off power change, one grid of subplots ---
    pooled_by_condition = {}
    for deg in stim_onset_degs:
        result = compute_pooled_tf_log_power_change(
            loaded_recordings, recordings_by_condition=recordings_by_condition, conditions=(deg,),
            win_s=win_s, step_s=step_s, smooth_window_s=smooth_window_s)
        if result is None:
            print(f"Skipping pooled Stim On vs Stim Off plot for {CONDITION_LABELS.get(deg, deg)}: not enough data.")
        else:
            pooled_by_condition[f"{CONDITION_LABELS.get(deg, deg)} ({deg} deg)"] = result

    if pooled_by_condition:
        plot_tf_power_change_grid(pooled_by_condition, ncols=3)

    plt.show()


if __name__ == "__main__":
    # analyse_spectrum(results_dir="results/CLAS_victor/victor_60_no_comp_20260817_144819")
    analyse_all(comparisons=(), use_csd=True)
