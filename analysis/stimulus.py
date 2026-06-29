"""
Stimulus reconstruction and edge-error analysis.

StimulusConfig is the single place to add new targeting rules — add fields here
and update from_yaml() and compute_reference_stimulus() accordingly.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class StimulusConfig:
    stim_onset_deg: float = 0.0
    stim_dur_deg:   float = 90.0
    stim_dur_unit:  str   = "deg"
    stim_dur_ms:    float = 20.0
    erp_latency_s:  float = 0.0

    @classmethod
    def from_yaml(cls, graph_config: dict) -> "StimulusConfig":
        opts = (
            graph_config.get("graph", {})
            .get("processors", {})
            .get("StimulusController", {})
            .get("options", {})
        )
        return cls(
            stim_onset_deg=opts.get("stim_onset_deg", 0.0),
            stim_dur_deg=opts.get("stim_dur_deg",   90.0),
            stim_dur_unit=opts.get("stim_dur_unit", "deg"),
            stim_dur_ms=opts.get("stim_dur_ms",     20.0),
            erp_latency_s=opts.get("erp_latency_s",  0.0),
        )


def get_edges(signal: np.ndarray, kind: str) -> np.ndarray:
    if kind == "rising":
        return np.where((signal[1:] == 1) & (signal[:-1] == 0))[0] + 1
    return np.where((signal[1:] == 0) & (signal[:-1] == 1))[0] + 1


def compute_reference_stimulus(
    phase: np.ndarray,
    iaf: np.ndarray,
    fs: float,
    config: StimulusConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the ideal stimulus signal from a reference phase time series,
    mirroring StimulusController::Process().

    Returns
    -------
    stim_ref      : binary ndarray, 1 inside the stimulus window
    onset_phases  : phase (rad) at each stimulus onset
    offset_phases : phase (rad) at each stimulus offset
    """
    iaf_safe  = np.where(np.isfinite(iaf) & (iaf > 0))[0]
    onset_rad = np.deg2rad(config.stim_onset_deg)

    if config.stim_dur_unit == "deg":
        stim_dur_rad = np.full(len(phase), np.deg2rad(config.stim_dur_deg))
    else:
        stim_dur_rad = np.zeros_like(phase)
        stim_dur_rad[iaf_safe] = (config.stim_dur_ms / 1000.0) * 2.0 * np.pi * iaf[iaf_safe]

    corrected_phase = phase.copy()
    corrected_phase[iaf_safe] = (
        phase[iaf_safe] + 2.0 * np.pi * iaf[iaf_safe] * config.erp_latency_s
    ) % (2.0 * np.pi)

    diff         = corrected_phase - onset_rad
    wrapped_diff = np.arctan2(np.sin(diff), np.cos(diff))

    stim_ref = np.zeros_like(phase, dtype=float)
    stim_ref[iaf_safe] = (np.abs(wrapped_diff[iaf_safe]) < stim_dur_rad[iaf_safe] / 2.0).astype(float)

    # Reset and rebuild with fixed duration per onset (matches real-time pipeline)
    rising   = get_edges(stim_ref, "rising")
    stim_ref = np.zeros_like(phase, dtype=float)
    for r in rising:
        dur_samples = int(round(stim_dur_rad[r] / (2 * np.pi * iaf[r]) * fs))
        stim_ref[r : r + dur_samples] = 1

    onset_phases  = phase[rising] if len(rising) else np.array([])
    offset_phases = onset_phases + stim_dur_rad[rising] if len(rising) else np.array([])
    return stim_ref, onset_phases, offset_phases


def compute_stimulus_edge_errors(
    ref_signal: np.ndarray,
    actual_signal: np.ndarray,
    time_us: np.ndarray,
    phase: np.ndarray,
    start_ts: float,
) -> tuple[dict | None, dict | None]:
    """
    Compare rising/falling edges of actual_signal against ref_signal.
    Edges are matched by nearest index; unmatched extras are discarded.

    Returns (onset_error, offset_error) dicts with:
        time_s  : time of the actual edge in seconds from start_ts
        values  : phase error in degrees (actual − reference)
    """
    n = min(len(ref_signal), len(actual_signal), len(time_us), len(phase))
    ref_signal    = ref_signal[:n]
    actual_signal = actual_signal[:n]
    time_us       = time_us[:n]
    phase         = phase[:n]

    def match_and_diff(ref_idx, act_idx, label):
        if len(ref_idx) == 0 or len(act_idx) == 0:
            return None
        if len(ref_idx) != len(act_idx):
            print(f"Warning: {label} edge count mismatch — "
                  f"ref={len(ref_idx)}, actual={len(act_idx)}. Matching by nearest index.")
        times, errs = [], []
        for i in act_idx:
            j   = int(np.argmin(np.abs(ref_idx - i)))
            err = float(np.angle(np.exp(1j * (phase[i] - phase[ref_idx[j]])), deg=True))
            times.append((time_us[i] - start_ts) / 1e6)
            errs.append(err)
        return {"time_s": np.array(times), "values": np.array(errs)}

    onset_err  = match_and_diff(get_edges(ref_signal,    "rising"),
                                get_edges(actual_signal, "rising"),  "onset")
    offset_err = match_and_diff(get_edges(ref_signal,    "falling"),
                                get_edges(actual_signal, "falling"), "offset")
    return onset_err, offset_err
