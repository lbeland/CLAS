"""
EEG pipeline analysis: exports signals to EDF and plots phase/frequency errors.

Usage:
    python plot_results.py

Outputs:
    pipeline_signals.edf   — all signals (raw, filtered, phase estimates, etc.)
    error_analysis.png     — error time series + histograms only
"""

import datetime
import pytz
import os
import yaml
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from scipy.stats import circmean, circstd
from read_output import get_signal_data
from pathlib import Path
import mne

mne.viz.set_browser_backend("matplotlib")
mne.viz.use_browser_backend("matplotlib")

# mpl.rcParams.update({
#     "text.usetex": True,
# })


# Automatic link to the last run results
RESULTS_DIR = "_last_run"

COMMON_BBOX = dict(
    facecolor="white",
    edgecolor="0.8",
    boxstyle="round,pad=0.2",
    alpha=0.85,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


def load_processor_signals(fs, results_dir, processors: list[str], timestamps: bool = True) -> dict:
    """Load raw signal data from all processors into a dict keyed by label."""
    samples = {}

    print(results_dir)
    for processor in processors:
        if processor == "Producer":
            # Data
            file = get_results_file(processor, slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(10)), timestamps=timestamps)
            if signal is None:
                continue
            for idx, channel in enumerate(signal.T):
                samples[f"{processor}_{idx}"] = {"x": time, "y": channel}
            # Metadata
            file = get_results_file(processor, name=".meta_out", slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(3)), timestamps=timestamps)
            if signal is None:
                continue
            for idx, channel in enumerate(signal.T):
                samples[f"{processor}_meta_{idx}"] = {"x": time, "y": channel}

        elif processor == "SourceClient":
            # Slot 0: primary EEG channels (up to 32)
            file = get_results_file(processor, slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(32)), timestamps=timestamps)
            if signal is None:
                continue
            for idx, channel in enumerate(signal.T):
                samples[f"SourceClient_{idx}"] = {"x": time, "y": channel}
            # Slot 1: AUX and trigger channels
            file = get_results_file(processor, slot=1, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(8)), timestamps=timestamps)
            if signal is not None:
                samples["SourceClient_AUX"] = {"x": time, "y": signal[:, 0] if signal.ndim > 1 else signal}

            file = get_results_file(processor, slot=1, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=8, timestamps=timestamps)
            if signal is not None:
                binary = (signal > 0.5).astype(float)
                signal = _trim_falling_edges(binary, fs, 11.0)
                samples["SourceClient_TRIGGER"] = {"x": time, "y": signal}

        elif processor == "PhaseEstimator":
            # Slot 0: phase estimates
            file = get_results_file(processor, slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
            if signal is not None:
                samples[f"{processor}_phase"] = {"x": time, "y": signal}
            # Slot 1: Real-part of analytic signal
            file = get_results_file(processor, slot=1, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
            if signal is not None:
                samples[f"{processor}_real"] = {"x": time, "y": signal}

        else:
            result_name = "ch_idx_out" if processor == "ChannelSelector" else ".out"
            file = get_results_file(processor, name=result_name, slot=0, results_dir=results_dir)
            if file is not None:
                signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
                if signal is not None:
                    samples[f"{processor}"] = {"x": time, "y": signal}

    first_timestamps = [samples[key]["x"][0] for key in samples]
    assert len(set(first_timestamps)) == 1, "Mismatched timestamps across processors"

    return samples


def extract_ground_truth(samples: dict) -> dict:
    """
    Extract ground truth signals from either SourceClient or Producer slots,
    plus any saved simulated_signal.npy file.
    Returns dict with keys: raw, time, true_phase, true_amplitude, true_inst_freq.
    """
    if "SourceClient_0" in samples:
        # From the 32 EEG channels, generate one "raw" signal that is build sample by sample with the signal that was selected by the ChannelSelector in real time
        # Because thats what the PhaseEstimator got as input
        if "ChannelSelector" in samples:
            channel_selected = samples["ChannelSelector"]["y"] - 1 # convert to zero-based

            raw = []
            for sample_idx, select_idx in enumerate(channel_selected):
                raw.append(samples[f"SourceClient_{select_idx}"]["y"][sample_idx])

            time  = samples["ChannelSelector"]["x"]
        else:
            raw = samples["SourceClient_0"]["y"]
            time = samples["SourceClient_0"]["x"]
            
        if os.path.exists("simulated_signal.npy"):
            loaded = np.load("simulated_signal.npy")
            assert raw[0, 0] == loaded["value"][0], \
                "Loaded simulated signal does not match SourceClient signal"
            return {
                "raw": raw, "time": time,
                "true_amplitude": loaded["amplitude"],
                "true_phase":     np.angle(np.exp(1j * loaded["phase"])),
                "true_inst_freq": loaded["inst_freq"],
            }
        return {"raw": raw, "time": time,
                "true_amplitude": None, "true_phase": None, "true_inst_freq": None}

    elif "Producer_0" in samples:
        if "ChannelSelector" in samples:
            channel_selected = samples["ChannelSelector"]["y"]

            raw = []
            for sample_idx, select_idx in enumerate(channel_selected):
                raw.append(samples[f"Producer_{select_idx-1}"]["y"][sample_idx])    # zero based

            raw = np.array(raw)
            time  = samples["ChannelSelector"]["x"]
        else:
            raw = samples["Producer_0"]["y"]
            time = samples["Producer_0"]["x"]

        return {
            "raw":            raw,
            "time":           time,
            "true_amplitude": samples.get("Producer_meta_0", {}).get("y"),
            "true_phase":     np.angle(np.exp(1j * samples["Producer_meta_1"]["y"])),
            "true_inst_freq": samples.get("Producer_meta_2", {}).get("y"),
        }

    return None


# ---------------------------------------------------------------------------
# Offline Hilbert reference
# ---------------------------------------------------------------------------

def compute_hilbert_reference(raw: np.ndarray, fs: float, f_low: float = 4.0, f_high: float = 8.0):
    """Bandpass + Hilbert to produce an offline phase reference."""
    sos = butter(1, [f_low, f_high], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, raw)
    analytic = hilbert(filtered)
    return filtered, np.angle(analytic)


# ---------------------------------------------------------------------------
# EDF helpers
# ---------------------------------------------------------------------------

def _trim_falling_edges(y: np.ndarray, fs: float, trim_ms: float) -> np.ndarray:
    """
    Shift each falling edge (1→0 transition) earlier by trim_ms milliseconds.
    Pulse onsets are preserved exactly; only the end of each pulse is shortened.
    Pulses shorter than trim_ms will have one single sample remaining.
    """
    trim_samples = int(round(trim_ms * fs / 1000))
    y = y.copy()
    # Falling edge index: last sample that is 1 before a 0
    falling = np.where((y[:-1] == 1) & (y[1:] == 0))[0]  # index of last 1
    # Raising edge index: last sample that is 0 before a 1 (used to avoid trimming onsets)
    raising = np.where((y[:-1] == 0) & (y[1:] == 1))[0]  # index of last 0
    assert len(falling) == len(raising) or len(falling) == len(raising) - 1
    for idx,f in enumerate(falling):
        start = max(raising[idx], f - trim_samples + 1)
        y[start : f + 1] = 0.0
    return y


# ---------------------------------------------------------------------------
# EDF export
# ---------------------------------------------------------------------------

def write_edf(
    filepath: str,
    fs: float,
    ground_truth: dict,
    samples: dict,
    hilbert_phase: np.ndarray,
    stim_ref: np.ndarray = None,
    filtered: np.ndarray = None,
) -> None:
    """Write all pipeline signals to an EDF file."""

    raw  = ground_truth["raw"]
    time = ground_truth["time"]
    n    = len(raw)

    # Build channel list: (label, data, physical_min, physical_max, dimension)
    channels = [("Raw", "eeg", raw)]

    # Filter channels
    if filtered is not None:
        channels.append(("Filt_off", "misc", filtered[:n]))

    if samples.get("GlobalFilter") is not None:
        bp = samples["GlobalFilter"]["y"]
        t_bp = samples["GlobalFilter"]["x"]
        channels.append(("Filt_on", "misc", bp[:n]))

    # Phase channels
    if hilbert_phase is not None:
        channels.append(("Hilbert_phi", "misc", hilbert_phase[:n]))

    if ground_truth["true_phase"] is not None:
        channels.append(("True_phi", "misc", ground_truth["true_phase"][:n]))

    if samples.get("PhaseEstimator_phase") is not None:
        phase_est = samples["PhaseEstimator_phase"]["y"]
        phase_est = np.nan_to_num(phase_est)   # replace any NaN with zero for EDF export
        channels.append(("Online_phi", "misc", phase_est[:n]))

    # Stimuli channels
    if stim_ref is not None:
        channels.append(("Target_Stim", "stim", stim_ref[:n]))

    if samples.get("StimulusController") is not None:
        y_st  = samples["StimulusController"]["y"]
        channels.append(("Stimulus", "stim", y_st[:n]))

    if samples.get("SourceClient_TRIGGER") is not None:
        y_st  = samples["SourceClient_TRIGGER"]["y"]
        channels.append(("Trigger", "stim", y_st[:n]))

    # AUX channel
    if samples.get("SourceClient_AUX") is not None:
        y_aux = samples["SourceClient_AUX"]["y"]
        channels.append(("AUX", "misc", y_aux[:n]))

    eeg_channels = [channel for channel in samples.keys() if channel.startswith("SourceClient_") and channel.split("_")[1].isdigit()]
    for ch in eeg_channels:
        ch_idx = ch.split("_")[1]
        y_ch = samples[ch]["y"]
        channels.append((f"EEG_{ch_idx}", "eeg", y_ch[:n]))

    # if ground_truth["true_amplitude"] is not None:
    #     channels.append(("True_amplitude", ground_truth["true_amplitude"][:n]))

    # IAF channels
    if ground_truth["true_inst_freq"] is not None:
        channels.append(("True_inst_freq_Hz", "misc",ground_truth["true_inst_freq"][:n]))

    if samples.get("IAFEstimator") is not None:
        iaf = samples["IAFEstimator"]["y"]
        iaf = np.nan_to_num(iaf)   # replace any NaN with zero for EDF export
        channels.append(("IAF_est_Hz", "misc", iaf[:n]))

    info = mne.create_info([ch[0] for ch in channels], sfreq=fs, ch_types=[ch[1] for ch in channels], verbose=False)
    raw = mne.io.RawArray([ch[2] for ch in channels], info, verbose=False)
    # Scale to volts
    raw.apply_function(lambda x: x * 1e-6, picks="eeg")  # EEG channels in microvolts → volts
    start_dt = datetime.datetime.fromtimestamp(time[0] / 1e6, tz=pytz.timezone("Europe/Berlin")).replace(tzinfo=datetime.timezone.utc)
    raw.set_meas_date(start_dt)
    raw.export(filepath, fmt="edf", add_ch_type=True, physical_range="channelwise", overwrite=True, verbose=False)

    print("EDF written.")


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------

def compute_errors(
    samples: dict,
    ground_truth: dict,
    hilbert_phase: np.ndarray,
    start_ts: float,
    fs: float,
    config: dict,
) -> list[dict]:
    """
    Return a list of error records, each with:
      label, time_s, error_values, unit
    """
    errors = []
    true_phase     = ground_truth["true_phase"]
    true_inst_freq = ground_truth["true_inst_freq"]
    raw            = ground_truth["raw"]
 
    # Phase error
    if samples.get("PhaseEstimator_phase") is not None:
        t   = (samples["PhaseEstimator_phase"]["x"] - start_ts) / 1e6
        phi = samples["PhaseEstimator_phase"]["y"]
        n   = len(phi)
 
        if true_phase is not None:
            err = np.angle(np.exp(1j * (true_phase[:n] - phi)), deg=True)
        else:
            err = np.angle(np.exp(1j * (hilbert_phase[:n] - phi)), deg=True)
 
        errors.append({"label": "Phase error", "time_s": t, "values": err, "unit": "degrees"})
 
        # Hilbert offline error (only when true phase known)
        if true_phase is not None:
            h_err = np.angle(np.exp(1j * (true_phase[:n] - hilbert_phase[:n])), deg=True)
            errors.append({"label": "Hilbert ref error", "time_s": t, "values": h_err,
                           "unit": "degrees", "linestyle": "--"})
 
    # IAF error
    if samples.get("IAFEstimator") is not None and true_inst_freq is not None:
        t   = (samples["IAFEstimator"]["x"] - start_ts) / 1e6
        iaf = samples["IAFEstimator"]["y"]
        n   = min(len(iaf), len(true_inst_freq))
        err = iaf[:n] - true_inst_freq[:n]
        errors.append({"label": "IAF error", "time_s": t[:n], "values": err, "unit": "Hz"})
 
    # Stimulus onset/offset errors: compare actual trigger edges against a
    # reference stimulus reconstructed from the offline/true phase.
    stim_ref = None
    if samples.get("StimulusController") is not None:
        if samples.get("SourceClient_TRIGGER") is not None:
            trigger_y = samples["SourceClient_TRIGGER"]["y"]
        elif samples.get("StimulusController") is not None:
            trigger_y = samples["StimulusController"]["y"]

        ref_phase = true_phase if true_phase is not None else hilbert_phase
        ref_label = "true" if true_phase is not None else "Hilbert"
 
        # IAF for phase-advance correction and dur_unit="ms" window scaling.
        # Priority: ground truth estimate → true IAF → constant 10 Hz fallback.
        if true_inst_freq is not None:
            iaf_y = true_inst_freq
        elif samples.get("IAFEstimator") is not None:
            iaf_y = samples["IAFEstimator"]["y"]
        else:
            print("No IAF information available; using constant 10 Hz for stimulus reconstruction")
            iaf_y = np.full(len(ground_truth["time"]), 10.0)
 
        stim_config = config.get("graph", {}).get("processors", {}).get("StimulusController", {}).get("options",{})
        stim_ref, _, _ = compute_reference_stimulus(
            phase=ref_phase,
            iaf=iaf_y,
            stim_onset_deg=stim_config.get("stim_onset_deg", 0.0),   # match TurboLinkCLAS.yaml
            stim_dur_deg=stim_config.get("stim_dur_deg", 90.0),      # match TurboLinkCLAS.yaml
            stim_dur_unit=stim_config.get("stim_dur_unit", "deg"),    # match TurboLinkCLAS.yaml
            stim_dur_ms=stim_config.get("stim_dur_ms", 20.0),       # fallback when IAF unavailable
            erp_latency_s=stim_config.get("erp_latency_s", 0.0),      # match TurboLinkCLAS.yaml
        )
 
        # Bring actual trigger onto the raw EEG timeline
        trigger_binary = (trigger_y > 0.5).astype(float)
 
        onset_err, offset_err = compute_stimulus_edge_errors(
            ref_signal=stim_ref,
            actual_signal=trigger_binary,
            time_us=ground_truth["time"],
            phase=ref_phase,
            start_ts=start_ts,
        )
 
        if onset_err is not None:
            errors.append({
                "label": f"Stim onset error",
                "time_s": onset_err["time_s"],
                "values": onset_err["values"],
                "unit": "degrees",
            })
        if offset_err is not None:
            errors.append({
                "label": f"Stim offset error",
                "time_s": offset_err["time_s"],
                "values": offset_err["values"],
                "unit": "degrees",
                "linestyle": "--",
            })
 
    return errors, stim_ref


# ---------------------------------------------------------------------------
# Stimulus reconstruction  (translated from StimulusController::Process())
# ---------------------------------------------------------------------------
 
def compute_reference_stimulus(
    phase: np.ndarray,
    iaf: np.ndarray,
    stim_onset_deg: float = 0.0,
    stim_dur_deg: float = 90.0,
    stim_dur_unit: str = "deg",
    stim_dur_ms: float = 20.0,
    erp_latency_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the ideal stimulus signal from a reference phase time series,
    mirroring StimulusController::Process().
 
    Audio and pipeline latencies are excluded (not available offline).
    Only erp_latency_s is applied, matching the yaml option of the same name.
 
    The stimulus window is open when:
        |wrap(corrected_phase - onset_rad)| < stim_dur_rad / 2
 
    where corrected_phase = phase + 2π * iaf * erp_latency_s.
 
    Parameters
    ----------
    phase         : instantaneous phase in radians, shape (N,)
    iaf           : instantaneous alpha frequency in Hz, shape (N,)
    time_us       : sample timestamps in microseconds, shape (N,)  [unused here,
                    kept for API symmetry with compute_errors]
    stim_onset_deg: target phase in degrees (yaml: stim_onset_deg)
    stim_dur_deg  : window width in degrees (yaml: stim_dur_deg)
    stim_dur_unit : "deg" or "ms"  (yaml: stim_dur_unit)
    stim_dur_ms   : window width in ms, used when stim_dur_unit=="ms" or as
                    fallback when IAF is unavailable  (yaml: stim_dur_ms)
    erp_latency_s : ERP latency in seconds  (yaml: erp_latency_s)
 
    Returns
    -------
    stim_ref      : binary ndarray, 1 inside the stimulus window
    onset_phases  : phase (rad) at each rising edge of stim_ref
    offset_phases : phase (rad) at each falling edge of stim_ref
    """
    stim_ref = np.zeros_like(phase, dtype=float)
    iaf_safe  = np.where(np.isfinite(iaf) & (iaf > 0))[0]
    onset_rad = np.deg2rad(stim_onset_deg)
 
    # stim_dur_rad — may vary sample-by-sample in "ms" mode
    if stim_dur_unit == "deg":
        stim_dur_rad = np.full(len(phase), np.deg2rad(stim_dur_deg))
    else:
        # "ms" mode: window width in radians scales with current IAF
        stim_dur_rad = np.zeros_like(phase)
        stim_dur_rad[iaf_safe] = (stim_dur_ms / 1000.0) * 2.0 * np.pi * iaf[iaf_safe]
 
    # Phase advance for ERP latency (audio/pipeline delay omitted — offline)
    corrected_phase = phase.copy()
    corrected_phase[iaf_safe] = (phase[iaf_safe] + 2.0 * np.pi * iaf[iaf_safe] * erp_latency_s) % (2.0 * np.pi)
 
    # Angular distance to onset target, wrapped to [-π, π]
    diff         = corrected_phase - onset_rad
    wrapped_diff = np.arctan2(np.sin(diff), np.cos(diff))
 
    stim_ref[iaf_safe] = (np.abs(wrapped_diff[iaf_safe]) < stim_dur_rad[iaf_safe] / 2.0).astype(float)
 
    rising  = np.where((stim_ref[1:] == 1) & (stim_ref[:-1] == 0))[0] + 1
    falling = np.where((stim_ref[1:] == 0) & (stim_ref[:-1] == 1))[0] + 1
 
    onset_phases  = phase[rising]  if len(rising)  else np.array([])
    offset_phases = onset_phases + stim_dur_rad[rising]
    # offset_phases = phase[falling] if len(falling) else np.array([])
 
    return stim_ref, onset_phases, offset_phases
 
 
def compute_stimulus_edge_errors(
    ref_signal: np.ndarray,
    actual_signal: np.ndarray,
    time_us: np.ndarray,
    phase: np.ndarray,
    start_ts: float,
) -> tuple[dict | None, dict | None]:
    """
    Compare rising/falling edges of the actual trigger against the reference
    stimulus. For each matched edge pair, compute the phase error
    (actual − reference) in degrees, wrapped to [−180, +180].
 
    Edges are matched by nearest-neighbour in time. If reference and actual
    have different numbers of edges a warning is printed; extra edges on
    either side are left unmatched and discarded.
 
    Returns two dicts (onset_error, offset_error), each with:
        time_s  : time of the *reference* edge in seconds from start_ts
        values  : phase error in degrees  (actual phase − reference phase)
    Returns None if no matched pairs exist for that edge type.
    """
    n             = min(len(ref_signal), len(actual_signal), len(time_us), len(phase))
    ref_signal    = ref_signal[:n]
    actual_signal = actual_signal[:n]
    t             = time_us[:n]
    ph            = phase[:n]
 
    def get_edges(signal, kind):
        if kind == "rising":
            idx = np.where((signal[1:] == 1) & (signal[:-1] == 0))[0] + 1
        else:
            idx = np.where((signal[1:] == 0) & (signal[:-1] == 1))[0] + 1
        return t[idx], ph[idx]
 
    ref_on_t,  ref_on_phi  = get_edges(ref_signal,    "rising")
    act_on_t,  act_on_phi  = get_edges(actual_signal, "rising")
    ref_off_t, ref_off_phi = get_edges(ref_signal,    "falling")
    act_off_t, act_off_phi = get_edges(actual_signal, "falling")
 
    def match_and_diff(ref_t, ref_phi, act_t, act_phi, label):
        if len(ref_t) == 0 or len(act_t) == 0:
            return None
        if len(ref_t) != len(act_t):
            print(f"Warning: {label} edge count mismatch — "
                  f"ref={len(ref_t)}, actual={len(act_t)}. Matching by nearest time.")
        times, errs = [], []
        for at, ap in zip(act_t, act_phi):
            j   = int(np.argmin(np.abs(ref_t - at)))
            err = float(np.angle(np.exp(1j * (ref_phi[j] - ap)), deg=True))
            times.append((at - start_ts) / 1e6)
            errs.append(err)
        # for rt, rp in zip(ref_t, ref_phi):
        #     j   = int(np.argmin(np.abs(act_t - rt)))
        #     err = float(np.angle(np.exp(1j * (rp - act_phi[j])), deg=True))
        #     times.append((rt - start_ts) / 1e6)
        #     errs.append(err)
        return {"time_s": np.array(times), "values": np.array(errs)}
 
    onset_err  = match_and_diff(ref_on_t,  ref_on_phi,  act_on_t,  act_on_phi,  "onset")
    offset_err = match_and_diff(ref_off_t, ref_off_phi, act_off_t, act_off_phi, "offset")
    return onset_err, offset_err


# ---------------------------------------------------------------------------
# Plotting (errors only)
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
    """
    Returns:
        mu, sd, plv, pli
    """
    phi = np.asarray(phi_rad, float)
    if phi.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    mu = float(circmean(phi, high=np.pi, low=-np.pi))
    sd = float(circstd(phi, high=np.pi, low=-np.pi))
    plv = float(np.abs(np.mean(np.exp(1j * phi))))
    pli = float(np.abs(np.mean(np.sign(np.sin(phi)))))

    return mu, sd, plv, pli

def _wrap_phase(phi):
    """Wrap phase to [-pi, pi]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def plot_errors(errors: list[dict], output_path: str = "error_analysis.png", time_range: tuple = None) -> None:
    """Plot error time series and histograms; save to output_path."""
    if not errors:
        print("No errors to plot.")
        return

    ax_ts = plt.figure(figsize=(10, 5)).add_subplot(111)
    fig_polars = plt.figure(figsize=(3 * len(errors), 3.5))
    fig_polars.suptitle("Error distributions", fontsize=12)
    ax_polars = [fig_polars.add_subplot(1, len(errors), i+1, projection="polar") for i in range(len(errors))]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    bin_width_deg = 10
    bin_width_rad = np.radians(bin_width_deg)
    bins = np.arange(-180, 181, bin_width_deg)
    centers = (np.radians(bins[:-1]) + np.radians(bins[1:])) / 2
    
    hists = []
    for i, err in enumerate(errors):
        val_deg = err["values"]  # in degree
        vals_deg = val_deg[~np.isnan(val_deg)]
        hist = np.histogram(vals_deg, bins=bins)[0] / max(1, vals_deg.size) * 100
        hists.append(hist)

    r_max = max([x.max() for x in hists]) * 1.05
    r_max = max(r_max, 5)
    r_ticks = [t for t in [10, 20, 30] if t < r_max]

    for i, err in enumerate(errors):
        color = colors[i % len(colors)]
        ls    = err.get("linestyle", "-")
        vals  = err["values"]   # in degree
        t     = err["time_s"]
        label = err["label"]
        unit  = err["unit"]

        # Time series
        ax_ts.plot(t, vals, linestyle=ls, linewidth=1.2, color=color,
                   label=f"{label} ({unit})", alpha=0.85)

        # Histogram
        if np.nansum(np.abs(vals)) == 0:
            continue
        vals_deg = vals[~np.isnan(vals)]
        vals_rad = np.radians(vals_deg)
        mu_u, sd_u, plv_u, pli_u = _circ_stats(vals_rad)

        ax_polars[i].bar(centers, hists[i], width=bin_width_rad, color=color, edgecolor="0", linewidth=0.75)
        ax_polars[i].plot([mu_u, mu_u], [0, r_max], color="0", linewidth=2)
        _style_polar_axis(ax_polars[i], r_max, r_ticks)
        ax_polars[i].text(
        0.5, 0.45,
        rf"${np.round(np.degrees(mu_u), 1) + 0.0:.1f}^\circ \pm {np.round(np.degrees(sd_u), 1):.1f}^\circ$",
        transform=ax_polars[i].transAxes, bbox=COMMON_BBOX, va="top", ha="center"
        )
        ax_polars[i].set_title(label, fontsize=10)
    # # Style
    # for ax in axes:
    #     ax.minorticks_on()
    #     ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    #     ax.legend(frameon=False, fontsize=8)

    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("Error")
    ax_ts.set_title("Error over time")

    # ax_hist.set_xlabel("Error value")
    # ax_hist.set_ylabel("Count")
    # ax_hist.set_title("Error distribution")

    if time_range is not None:
        ax_ts.set_xlim(time_range)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Error plot saved: {output_path}")

def plot_spectrum(raw, samples: dict, fs: float) -> None:
    """Plot example spectrum of raw and filtered signals."""
    raw_spectrum = np.fft.rfft(raw)
    freqs = np.fft.rfftfreq(len(raw), d=1/fs)

    # freqs, raw_spectrum = welch(raw, fs=fs, nperseg=fs*5)  # Welch PSD estimate for smoother spectrum

    if samples.get("GlobalFilter") is not None:
        filt = samples["GlobalFilter"]["y"]
        filt_spectrum = np.fft.rfft(filt)
        # freqs, filt_spectrum = welch(samples["GlobalFilter"]["y"], fs=fs, nperseg=fs*5)
    else:
        filt_spectrum = None

    plt.figure(figsize=(8, 4))
    # plt.hist(freqs, bins=2*fs, weights=np.abs(raw_spectrum), alpha=0.7, label="Raw", color="blue")
    plt.plot(freqs, np.abs(raw_spectrum), label="Raw", alpha=0.7, color="blue")
    # plt.hist(freqs, bins=2*fs, weights=np.abs(filt_spectrum), alpha=0.7, label="Filtered", color="orange") if filt_spectrum is not None else None
    plt.plot(freqs, np.abs(filt_spectrum), label="Filtered", alpha=0.7, color="orange") if filt_spectrum is not None else None

    plt.xlim(0, 100)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Spectrum")
    plt.legend(facecolor="white", frameon=True)

def plot_time_series(ground_truth, samples: dict, hilbert_phase, stim_ref, fs: float, time_range: tuple = None) -> None:
    time_s = (ground_truth["time"] - ground_truth["time"][0]) / 1e6  # convert to seconds from start

    def _pick(label: str):
        for signal in samples.keys():
            if signal == label:
                return samples[signal]["y"]
        return None

    raw_eeg = ground_truth["raw"]-np.mean(ground_truth["raw"])
    filt_online = _pick("GlobalFilter")

    online_phase = _pick("PhaseEstimator_phase")
    true_phase = ground_truth["true_phase"] if ground_truth["true_phase"] is not None else None

    stimulus = _pick("StimulusController")
    trigger = _pick("SourceClient_TRIGGER")
    aux = _pick("SourceClient_AUX")
    if aux is not None:
        # Scale to be between 0 and 1 for plotting, and zero-centre
        aux = (aux - np.min(aux)) / (np.max(aux) - np.min(aux))
        aux = aux - np.mean(aux)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 9), height_ratios=[3, 1])
    ax_raw, ax_phase = axes

    if raw_eeg is not None:
        ax_raw.plot(time_s, raw_eeg, color="0.6", linewidth=0.8, label="Raw EEG")
    if filt_online is not None:
        ax_raw.plot(time_s, filt_online, color="0.3", linewidth=1.5, label="Filtered EEG (online)")
    if stim_ref is not None:
        # ax_raw.fill_between(time_s, stim_ref, color="tab:red", alpha=0.3, label="Target stim")
        ax_raw.fill_between(time_s, 0, 1, where=stim_ref,
                        color='tab:blue', alpha=0.4, transform=ax_raw.get_xaxis_transform(), label="Target stim")
    if trigger is not None:
        ax_raw.fill_between(time_s, 0, 1, where=trigger,
                        color='tab:orange', alpha=0.4, transform=ax_raw.get_xaxis_transform(), label="Stim")
    elif stimulus is not None:
        ax_raw.fill_between(time_s, 0, 1, where=stimulus,
                        color='tab:orange', alpha=0.4, transform=ax_raw.get_xaxis_transform(), label="Stimulus")
    ax_raw.set_ylabel("Amplitude (µV)")
    ax_raw.set_title("Raw and Filtered EEG with Stimuli")
    ax_raw.set_ylim(-150,100)
    ax_raw.legend(facecolor="white", frameon=True, fontsize=8, loc="upper right")

    if hilbert_phase is not None:
        ax_phase.plot(time_s, hilbert_phase, color="tab:blue", linewidth=1.5, label="Offline Hilbert phase")
    if online_phase is not None:
        ax_phase.plot(time_s, online_phase, color="tab:orange", linewidth=1.5, label="Online phase")
    if true_phase is not None:
        ax_phase.plot(time_s, true_phase, color="tab:brown", linewidth=1.0, label="True phase")
    ax_phase.set_ylabel("Phase (rad)")
    ax_phase.set_title("Phase Estimates")
    ax_phase.legend(facecolor="white", frameon=True, fontsize=8, loc="upper right")
    ax_phase.set_xlabel("Time (s)")

    # if stim_ref is not None:
    #     ax_stim.step(time_s, stim_ref, where="post", color="tab:red", linewidth=1.2, label="Target stim")
    # # if stimulus is not None:
    # #     ax_stim.step(time_s, stimulus, where="post", color="tab:olive", linewidth=1.1, alpha=0.9, label="Stimulus")
    # if trigger is not None:
    #     ax_stim.step(time_s, trigger, where="post", color="tab:purple", linewidth=1.0, alpha=0.9, label="Stim")
    # if aux is not None:
    #     ax_stim.plot(time_s, aux, color="tab:orange", linewidth=1, alpha=0.5, label="AUX")
    # ax_stim.set_ylabel("State")
    # ax_stim.set_xlabel("Time (s)")
    # ax_stim.set_title("Stimulus Signals")
    # ax_stim.set_ylim(-0.1, 1.1)
    # ax_stim.legend(frameon=False, fontsize=8, loc="upper right")

    ax_raw.set_xlim(time_range) if time_range is not None else None

    for ax in axes:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.savefig("time_series.svg", dpi=300, bbox_inches="tight")

    return



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_pipeline(
    f0: float,
    results_dir: str,
    plot_path: str = "error_analysis.svg",
) -> None:

    graph_file = glob.glob(os.path.join(results_dir, "*.yaml"))
    print(results_dir)

    with open(graph_file[0], "r") as f:
        graph_config = yaml.safe_load(f)

    fs = graph_config.get("graph", {}).get("defaults", {}).get("fs", None)

    processors = graph_config.get("graph", {}).get("processors", [])
    if processors is None:
        raise ValueError(f"No processors found in graph config.")

    # 1. Load raw processor outputs
    samples = load_processor_signals(fs, results_dir, processors)

    # 2. Identify ground-truth signals
    ground_truth = extract_ground_truth(samples)
    if ground_truth is None:
        print("Error: no source signal found (SourceClient or Producer). Aborting.")
        return

    # 3. Offline Hilbert reference
    raw = ground_truth["raw"]
    filtered, hilbert_phase = compute_hilbert_reference(raw, fs, f_low=f0-2.0, f_high=f0+2.0)

    # 4. Compute errors
    start_ts = ground_truth["time"][0]
    errors, stim_ref = compute_errors(samples, ground_truth, hilbert_phase, start_ts, fs, graph_config)

    # # 5. Export to EDF
    edf_path = os.path.join(results_dir, os.path.basename(results_dir) + ".edf")
    write_edf(edf_path, fs, ground_truth, samples, hilbert_phase, stim_ref, filtered)

    time_range_to_plot = (17,18)  # seconds, relative to start of recording

    # 6. Plot errors
    plot_errors(errors, output_path=plot_path)

    # 7. Plot example spectrum of filtered data
    # plot_spectrum(raw, samples, fs)

    # 8. Plot time series from EDF
    plot_time_series(ground_truth, samples, hilbert_phase, stim_ref, fs, time_range=time_range_to_plot)
    # plot_errors(errors, output_path=plot_path, time_range=time_range_to_plot)

    plt.show()


if __name__ == "__main__":

    analyse_pipeline(f0=10,results_dir="results/eike_20260529_1600")

    # _last_run
    # eike_20260529_1600