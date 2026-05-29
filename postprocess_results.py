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
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from read_output import get_signal_data

# Optional EDF support — install with: pip install pyedflib
try:
    import pyedflib
    HAS_EDF = True
except ImportError:
    HAS_EDF = False
    print("Warning: pyedflib not installed. EDF export disabled. Install with: pip install pyedflib")

RESULTS_DIR = "rt_c_results"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_results_file(processor_name: str, slot: int = 0, results_dir=None) -> str | None:
    """Return path to the binary results file for a given processor and slot."""
    dir = results_dir if results_dir is not None else RESULTS_DIR
    for filename in os.listdir(dir):
        if filename.endswith(f"{slot}.bin") and processor_name in filename:
            return os.path.join(dir, filename)
    return None


def load_processor_signals(processors: list[str], timestamps: bool = True, results_dir=None) -> dict:
    """Load raw signal data from all processors into a dict keyed by label."""
    samples = {}

    for processor in processors:
        if processor == "Producer":
            for slot in range(4):
                signal, time = get_signal_data(get_results_file(processor, slot, results_dir), timestamps=timestamps)
                if signal is not None:
                    samples[f"{processor}_{slot}"] = {"x": time, "y": signal}

        elif processor == "SourceClient":
            # Slot 0: primary EEG channel
            signal, time = get_signal_data(get_results_file(processor, 0, results_dir), channel=0, timestamps=timestamps)
            if signal is not None:
                samples["SourceClient_0"] = {"x": time, "y": signal}
            # Slot 1: AUX and trigger channels
            for label, channel in [("AUX", 0), ("TRIGGER", 8)]:
                signal, time = get_signal_data(get_results_file(processor, 1, results_dir), channel=channel, timestamps=timestamps)
                if signal is not None:
                    samples[f"SourceClient_1_{label}"] = {"x": time, "y": signal}

        elif processor == "PhaseEstimator":
            for slot in range(2):
                signal, time = get_signal_data(get_results_file(processor, slot, results_dir), timestamps=timestamps)
                if signal is not None:
                    samples[f"{processor}_{slot}"] = {"x": time, "y": signal}

        else:
            file = get_results_file(processor, 0, results_dir)
            if file is not None:
                signal, time = get_signal_data(file, timestamps=timestamps)
            if signal is not None:
                samples[f"{processor}_0"] = {"x": time, "y": signal}

    return samples


def extract_ground_truth(samples: dict) -> dict:
    """
    Extract ground truth signals from either SourceClient or Producer slots,
    plus any saved simulated_signal.npy file.
    Returns dict with keys: raw, time, true_phase, true_amplitude, true_inst_freq.
    """
    if "SourceClient_0" in samples:
        raw   = samples["SourceClient_0"]["y"]
        time  = samples["SourceClient_0"]["x"]
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
        return {
            "raw":            samples["Producer_0"]["y"],
            "time":           samples["Producer_0"]["x"],
            "true_amplitude": samples.get("Producer_1", {}).get("y"),
            "true_phase":     np.angle(np.exp(1j * samples["Producer_2"]["y"]))
                              if "Producer_2" in samples else None,
            "true_inst_freq": samples.get("Producer_3", {}).get("y"),
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
    if not HAS_EDF:
        print("Skipping EDF export (pyedflib not available).")
        return

    raw  = ground_truth["raw"].squeeze()
    time = ground_truth["time"]
    n    = len(raw)

    # Build channel list: (label, data, physical_min, physical_max, dimension)
    channels = [("EEG_raw", raw)]

    # Filter channels
    if filtered is not None:
        channels.append(("EEG_filt_offline", filtered[:n]))

    if samples.get("GlobalFilter_0") is not None:
        bp = samples["GlobalFilter_0"]["y"].squeeze()
        t_bp = samples["GlobalFilter_0"]["x"]
        if len(bp) != n:
            bp = np.interp(time, t_bp, bp)
        channels.append(("EEG_filt_online", bp[:n]))

    # Phase channels
    if hilbert_phase is not None:
        channels.append(("Hilbert_phase_rad", hilbert_phase[:n]))

    if ground_truth["true_phase"] is not None:
        channels.append(("True_phase_rad", ground_truth["true_phase"][:n]))

    if samples.get("PhaseEstimator_0") is not None:
        phase_est = samples["PhaseEstimator_0"]["y"].squeeze()
        t_est = samples["PhaseEstimator_0"]["x"]
        if len(phase_est) != n:
            phase_est = np.interp(time, t_est, phase_est)
        channels.append(("Online_phase_rad", phase_est[:n]))

    # Stimuli channels
    if stim_ref is not None:
        channels.append(("Target_Stim", stim_ref[:n]))

    for key, label, ann_label, trim_ms in [
        ("StimulusController_0",   "Stimulus", "Stimulus", 0.0),
        ("SourceClient_1_TRIGGER", "Trigger",  "Trigger",  11.0),
    ]:
        if samples.get(key) is not None:
            t_ev  = samples[key]["x"]
            y_ev  = samples[key]["y"].squeeze()
            # Resample onto raw timeline as a continuous 0/1 channel
            binary = np.interp(time, t_ev, y_ev.astype(float))
            binary = (binary > 0.5).astype(float)   # re-binarise after interp
            if trim_ms > 0.0:
                binary = _trim_falling_edges(binary, fs, trim_ms)
            channels.append((label, binary[:n]))

    # AUX channel (raw, interpolated onto the EEG timeline)
    if samples.get("SourceClient_1_AUX") is not None:
        t_aux = samples["SourceClient_1_AUX"]["x"]
        y_aux = samples["SourceClient_1_AUX"]["y"].squeeze()
        if len(y_aux) != n:
            y_aux = np.interp(time, t_aux, y_aux)
        channels.append(("AUX_raw", y_aux[:n]))


    # if ground_truth["true_amplitude"] is not None:
    #     channels.append(("True_amplitude", ground_truth["true_amplitude"][:n]))

    # IAF channels
    if ground_truth["true_inst_freq"] is not None:
        channels.append(("True_inst_freq_Hz", ground_truth["true_inst_freq"][:n]))

    if samples.get("IAFEstimator_0") is not None:
        iaf = samples["IAFEstimator_0"]["y"].squeeze()
        t_iaf = samples["IAFEstimator_0"]["x"]
        if len(iaf) != n:
            iaf = np.interp(time, t_iaf, iaf)
        channels.append(("IAF_est_Hz", iaf[:n]))

    with pyedflib.EdfWriter(filepath, len(channels), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        start_dt = datetime.datetime.fromtimestamp(time[0] / 1e6, tz=pytz.timezone("Europe/Berlin"))
        f.setStartdatetime(start_dt)
        for i, (label, data) in enumerate(channels):
            data = data.astype(np.float64)
            pmin, pmax = float(np.nanmin(data)) - 1, float(np.nanmax(data)) + 1
            if pmin == pmax:           # flat signal / all-zeros edge-case
                pmax = pmin + 1.0
            f.setSignalHeader(i, {
                "label":            label,
                "dimension":        "",
                "sample_frequency": fs,
                "physical_min":     pmin,
                "physical_max":     pmax,
                "digital_min":      -32768,
                "digital_max":      32767,
                "transducer":       "",
                "prefilter":        "",
            })
        f.writeSamples([data.astype(np.float64) for _, data in channels])
        # for onset, duration, ann_label in annotations:
        #     f.writeAnnotation(onset, duration, ann_label)
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
    raw            = ground_truth["raw"].squeeze()
 
    # Phase error
    if samples.get("PhaseEstimator_0") is not None:
        t   = (samples["PhaseEstimator_0"]["x"] - start_ts) / 1e6
        phi = samples["PhaseEstimator_0"]["y"].squeeze()
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
    if samples.get("IAFEstimator_0") is not None and true_inst_freq is not None:
        t   = (samples["IAFEstimator_0"]["x"] - start_ts) / 1e6
        iaf = samples["IAFEstimator_0"]["y"].squeeze()
        n   = min(len(iaf), len(true_inst_freq))
        err = iaf[:n] - true_inst_freq[:n]
        errors.append({"label": "IAF error", "time_s": t[:n], "values": err, "unit": "Hz"})
 
    # Stimulus onset/offset errors: compare actual trigger edges against a
    # reference stimulus reconstructed from the offline/true phase.
    stim_ref = None
    if samples.get("StimulusController_0") is not None:
        if samples.get("SourceClient_1_TRIGGER") is not None:
            trigger_source = "measured"
            trigger_t = samples["SourceClient_1_TRIGGER"]["x"]
            trigger_y = samples["SourceClient_1_TRIGGER"]["y"].squeeze()
        elif samples.get("StimulusController_0") is not None:
            trigger_source = "internal"
            trigger_t = samples["StimulusController_0"]["x"]
            trigger_y = samples["StimulusController_0"]["y"].squeeze()

        ref_phase = true_phase if true_phase is not None else hilbert_phase
        ref_label = "true" if true_phase is not None else "Hilbert"
 
        # IAF for phase-advance correction and dur_unit="ms" window scaling.
        # Priority: ground truth estimate → true IAF → constant 10 Hz fallback.
        if true_inst_freq is not None:
            iaf_interp = true_inst_freq
        elif samples.get("IAFEstimator_0") is not None:
            iaf_t = samples["IAFEstimator_0"]["x"]
            iaf_y = samples["IAFEstimator_0"]["y"].squeeze()
            iaf_interp = np.interp(ground_truth["time"], iaf_t, iaf_y)
        else:
            iaf_interp = np.full(len(ground_truth["time"]), 10.0)
 
        stim_config = config.get("graph", {}).get("processors", {}).get("StimulusController", {}).get("options",{})
        stim_ref, _, _ = compute_reference_stimulus(
            phase=ref_phase,
            iaf=iaf_interp,
            time_us=ground_truth["time"],
            stim_onset_deg=stim_config.get("stim_onset_deg", 0.0),   # match TurboLinkCLAS.yaml
            stim_dur_deg=stim_config.get("stim_dur_deg", 90.0),      # match TurboLinkCLAS.yaml
            stim_dur_unit=stim_config.get("stim_dur_unit", "deg"),    # match TurboLinkCLAS.yaml
            stim_dur_ms=stim_config.get("stim_dur_ms", 20.0),       # fallback when IAF unavailable
            erp_latency_s=stim_config.get("erp_latency_s", 0.0),      # match TurboLinkCLAS.yaml
        )
 
        # Bring actual trigger onto the raw EEG timeline
        trigger_binary = np.interp(ground_truth["time"], trigger_t, trigger_y.astype(float))
        trigger_binary = (trigger_binary > 0.5).astype(float)
        if trigger_source == "measured":
            # Trim falling edges to correct for any stimulus duration (onset timing preserved)
            trigger_binary = _trim_falling_edges(trigger_binary, fs=fs, trim_ms=11.0)
 
        onset_err, offset_err = compute_stimulus_edge_errors(
            ref_signal=stim_ref,
            actual_signal=trigger_binary,
            time_us=ground_truth["time"],
            phase=ref_phase,
            start_ts=start_ts,
        )
 
        if onset_err is not None:
            errors.append({
                "label": f"Stim onset error ({ref_label} ref)",
                "time_s": onset_err["time_s"],
                "values": onset_err["values"],
                "unit": "degrees",
            })
        if offset_err is not None:
            errors.append({
                "label": f"Stim offset error ({ref_label} ref)",
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
    time_us: np.ndarray,
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
    offset_phases = phase[falling] if len(falling) else np.array([])
 
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
        for rt, rp in zip(ref_t, ref_phi):
            j   = int(np.argmin(np.abs(act_t - rt)))
            err = float(np.angle(np.exp(1j * (act_phi[j] - rp)), deg=True))
            times.append((rt - start_ts) / 1e6)
            errs.append(err)
        return {"time_s": np.array(times), "values": np.array(errs)}
 
    onset_err  = match_and_diff(ref_on_t,  ref_on_phi,  act_on_t,  act_on_phi,  "onset")
    offset_err = match_and_diff(ref_off_t, ref_off_phi, act_off_t, act_off_phi, "offset")
    return onset_err, offset_err


# ---------------------------------------------------------------------------
# Plotting (errors only)
# ---------------------------------------------------------------------------

def plot_errors(errors: list[dict], output_path: str = "error_analysis.png") -> None:
    """Plot error time series and histograms; save to output_path."""
    if not errors:
        print("No errors to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    ax_ts, ax_hist = axes

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, err in enumerate(errors):
        color = colors[i % len(colors)]
        ls    = err.get("linestyle", "-")
        vals  = err["values"]
        t     = err["time_s"]
        label = err["label"]
        unit  = err["unit"]

        # Time series
        ax_ts.plot(t, vals, linestyle=ls, linewidth=1.2, color=color,
                   label=f"{label} ({unit})", alpha=0.85)

        # Histogram
        if np.nansum(np.abs(vals)) == 0:
            continue
        mean, median, std = np.nanmean(vals), np.nanmedian(vals), np.nanstd(vals)
        ax_hist.hist(vals, bins="auto", alpha=0.65, edgecolor="black",
                     linewidth=0.4, color=color, label=f"{label}")
        ax_hist.axvline(mean,            linestyle="--", linewidth=1.2, color=color,
                        label=f"μ={mean:.2f}  σ={std:.2f}  med={median:.2f}")
        ax_hist.axvline(mean + std,      linestyle="-.", linewidth=0.9, color=color, alpha=0.7)
        ax_hist.axvline(mean - std,      linestyle="-.", linewidth=0.9, color=color, alpha=0.7)

    # Style
    for ax in axes:
        ax.minorticks_on()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=8)

    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("Error")
    ax_ts.set_title("Error over time")

    ax_hist.set_xlabel("Error value")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Error distribution")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Error plot saved: {output_path}")

def plot_spectrum(raw, samples: dict, fs: float) -> None:
    """Plot example spectrum of raw and filtered signals."""
    raw_spectrum = np.fft.rfft(raw)
    freqs = np.fft.rfftfreq(len(raw), d=1/fs)

    # freqs, raw_spectrum = welch(raw, fs=fs, nperseg=fs*5)  # Welch PSD estimate for smoother spectrum

    if samples.get("GlobalFilter_0") is not None:
        filt = samples["GlobalFilter_0"]["y"].squeeze()
        filt_spectrum = np.fft.rfft(filt)
        # freqs, filt_spectrum = welch(samples["GlobalFilter_0"]["y"].squeeze(), fs=fs, nperseg=fs*5)
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
    plt.legend()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_pipeline(
    f0: float,
    graph_config: dict,
    edf_path: str = "pipeline_signals.edf",
    plot_path: str = "error_analysis.png",
    results_dir: str = RESULTS_DIR,
) -> None:
    fs = graph_config.get("graph", {}).get("defaults", {}).get("fs", None)

    processors = graph_config.get("graph", {}).get("processors", [])
    if processors is None:
        raise ValueError(f"No processors found in graph config.")

    # 1. Load raw processor outputs
    samples = load_processor_signals(processors, results_dir=results_dir)

    # 2. Identify ground-truth signals
    ground_truth = extract_ground_truth(samples)
    if ground_truth is None:
        print("Error: no source signal found (SourceClient_0 or Producer_0). Aborting.")
        return

    # 3. Offline Hilbert reference
    raw = ground_truth["raw"].squeeze()
    _, hilbert_phase = compute_hilbert_reference(raw, fs, f_low=f0-2.0, f_high=f0+2.0)

    # 4. Compute errors
    start_ts = ground_truth["time"][0]
    errors, stim_ref = compute_errors(samples, ground_truth, hilbert_phase, start_ts, fs, graph_config)


    # 5. Export to EDF
    write_edf(edf_path, fs, ground_truth, samples, hilbert_phase, stim_ref)

    # 6. Plot errors
    plot_errors(errors, output_path=plot_path)

    # 7. Plot example spectrum of filtered data
    plot_spectrum(raw, samples, fs)

    plt.show()


if __name__ == "__main__":
    with open("resources/graphs/TurboLinkCLAS.yaml", "r") as f:
        graph_config = yaml.safe_load(f)
    analyse_pipeline(
        f0=6,
        graph_config=graph_config,
        results_dir="rt_c_results",
    )