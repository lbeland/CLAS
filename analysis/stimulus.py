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
            .get("StimControl", {})
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
    mirroring StimControl::Process().

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
    wrapped_diff = np.mod(diff, 2.0 * np.pi)

    stim_ref = np.zeros_like(phase, dtype=float)
    last_stim_idx = 0
    # Minimum needed distance between stim onset is 1 cycle of the current IAF + 10% buffer
    for i in iaf_safe:
        min_dist_stim = 1 / (iaf[i] + iaf[i] * 0.1)
        if wrapped_diff[i] < 0.1 * 2.0 * np.pi:  # 10% of an alpha cycle
            if (i - last_stim_idx) / fs >= min_dist_stim:
                stim_ref[i] = 1
                last_stim_idx = i
        
    # stim_ref[iaf_safe] = (np.abs(wrapped_diff[iaf_safe]) < stim_dur_rad[iaf_safe] / 2.0).astype(float)

    # Reset and rebuild with fixed duration per onset (matches real-time pipeline)
    rising   = get_edges(stim_ref, "rising")
    stim_ref = np.zeros_like(phase, dtype=float)
    for r in rising:
        dur_samples = int(round(stim_dur_rad[r] / (2 * np.pi * iaf[r]) * fs))
        stim_ref[r : r + dur_samples] = 1

    onset_phases  = phase[rising] if len(rising) else np.array([])
    offset_phases = onset_phases + stim_dur_rad[rising] if len(rising) else np.array([])
    return stim_ref, onset_phases, offset_phases


def _edge_phase_errors(
    phase: np.ndarray,
    iaf: np.ndarray,
    edge_idx: np.ndarray,
    target_rad,
    erp_latency_s: float,
    time_us: np.ndarray,
    start_ts: float,
) -> "dict | None":
    """Wrapped phase error (degrees) at each sample in ``edge_idx``: the
    erp-latency-corrected reference ``phase`` there minus ``target_rad`` (a
    scalar or an array aligned with ``edge_idx``). Sign: actual - target.

    Edges whose result is non-finite are dropped -- e.g. a sparse ``phase``
    that is NaN away from its evaluation points, or a NaN IAF feeding an
    ``target_rad`` array. Returns None if nothing survives.
    """
    edge_idx = np.asarray(edge_idx, dtype=int)
    if edge_idx.size == 0:
        return None
    # nan_to_num on the correction term only: a missing IAF should not by
    # itself discard an onset when erp_latency_s is 0 (0 * nan == nan).
    corr = 2.0 * np.pi * erp_latency_s * np.nan_to_num(iaf[edge_idx], nan=0.0)
    vals = np.angle(np.exp(1j * (phase[edge_idx] + corr - target_rad)), deg=True)
    keep = np.isfinite(vals)
    if not keep.any():
        return None
    return {"time_s": (time_us[edge_idx][keep] - start_ts) / 1e6,
            "values": np.asarray(vals)[keep]}


def compute_stimulus_edge_errors(
    trigger_binary: np.ndarray,
    time_us: np.ndarray,
    phase: np.ndarray,
    iaf: np.ndarray,
    config: StimulusConfig,
    start_ts: float,
) -> tuple[dict | None, dict | None]:
    """Phase error at each stimulus trigger edge, measured directly against the
    phase the controller targets.

    For every rising edge of ``trigger_binary`` the error is the wrapped
    difference between the erp-latency-corrected reference ``phase`` at that
    sample and ``stim_onset_deg``; for every falling edge it is measured
    against ``stim_onset_deg`` plus the nominal burst length -- ``stim_dur_deg``
    when ``stim_dur_unit == "deg"``, else ``2*pi*iaf*stim_dur_ms/1000`` (this
    mirrors ``compute_reference_stimulus``, whose reconstructed edges sit at
    exactly those target phases). Sign: actual - target, positive == overshoot.

    Returns (onset_error, offset_error) dicts with ``time_s`` / ``values``
    (degrees), or None for an edge kind with nothing to score.
    """
    n = min(len(trigger_binary), len(time_us), len(phase), len(iaf))
    trigger_binary = np.asarray(trigger_binary)[:n]
    time_us        = np.asarray(time_us)[:n]
    phase          = np.asarray(phase, dtype=float)[:n]
    iaf            = np.asarray(iaf, dtype=float)[:n]

    onset_rad = np.deg2rad(config.stim_onset_deg)
    rising    = get_edges(trigger_binary, "rising")
    falling   = get_edges(trigger_binary, "falling")

    if config.stim_dur_unit == "deg":
        dur_rad = np.deg2rad(config.stim_dur_deg)
    elif falling.size:
        dur_rad = 2.0 * np.pi * iaf[falling] * (config.stim_dur_ms / 1000.0)
    else:
        dur_rad = 0.0

    onset_err  = _edge_phase_errors(phase, iaf, rising,  onset_rad,
                                    config.erp_latency_s, time_us, start_ts)
    offset_err = _edge_phase_errors(phase, iaf, falling, onset_rad + dur_rad,
                                    config.erp_latency_s, time_us, start_ts)
    return onset_err, offset_err
