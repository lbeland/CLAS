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
    f0: np.ndarray,
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
    f0_safe  = np.where(np.isfinite(f0) & (f0 > 0))[0]
    onset_rad = np.deg2rad(config.stim_onset_deg)

    if config.stim_dur_unit == "deg":
        stim_dur_rad = np.full(len(phase), np.deg2rad(config.stim_dur_deg))
    else:
        stim_dur_rad = np.zeros_like(phase)
        stim_dur_rad[f0_safe] = (config.stim_dur_ms / 1000.0) * 2.0 * np.pi * f0[f0_safe]

    corrected_phase = phase.copy()
    corrected_phase[f0_safe] = (
        phase[f0_safe] + 2.0 * np.pi * f0[f0_safe] * config.erp_latency_s
    ) % (2.0 * np.pi)

    diff         = corrected_phase - onset_rad
    wrapped_diff = np.mod(diff, 2.0 * np.pi)

    stim_ref = np.zeros_like(phase, dtype=float)
    last_stim_idx = 0
    # Minimum needed distance between stim onset is 1 cycle of the current f0 + 10% buffer
    for i in f0_safe:
        min_dist_stim = 1 / (f0[i] + f0[i] * 0.1)
        if wrapped_diff[i] < 0.1 * 2.0 * np.pi:  # 10% of an alpha cycle
            if (i - last_stim_idx) / fs >= min_dist_stim:
                stim_ref[i] = 1
                last_stim_idx = i
        
    # stim_ref[f0_safe] = (np.abs(wrapped_diff[f0_safe]) < stim_dur_rad[f0_safe] / 2.0).astype(float)

    # Reset and rebuild with fixed duration per onset (matches real-time pipeline)
    rising   = get_edges(stim_ref, "rising")
    stim_ref = np.zeros_like(phase, dtype=float)
    for r in rising:
        dur_samples = int(round(stim_dur_rad[r] / (2 * np.pi * f0[r]) * fs))
        stim_ref[r : r + dur_samples] = 1

    onset_phases  = phase[rising] if len(rising) else np.array([])
    offset_phases = onset_phases + stim_dur_rad[rising] if len(rising) else np.array([])
    return stim_ref, onset_phases, offset_phases


def _edge_phase_errors(
    phase: np.ndarray,
    edge_idx: np.ndarray,
    target_rad: float,
    latency_samples: int,
    time_us: np.ndarray,
    start_ts: float,
) -> "dict | None":
    """Wrapped phase error (degrees) at each sample in ``edge_idx``, offset by
    ``latency_samples``: the reference ``phase`` there minus ``target_rad``.
    Sign: actual - target.

    ``latency_samples`` (the ERP latency in samples) is applied by reading
    ``phase`` that many samples *after* each edge, rather than analytically
    advancing ``phase[edge_idx]`` by an assumed constant angular velocity
    (``2*pi*f0*erp_latency_s``) -- this uses whatever the reference phase
    series actually did over that window instead of a linear approximation
    of it, so it stays accurate even when f0 is off or phase isn't advancing
    linearly there (e.g. online-estimator noise).

    Edges whose shifted index would run outside ``phase``, or whose result is
    non-finite (e.g. a sparse ``phase`` that is NaN away from its evaluation
    points), are dropped. Returns None if nothing survives.
    """
    edge_idx = np.asarray(edge_idx, dtype=int)
    if edge_idx.size == 0:
        return None
    eval_idx  = edge_idx + latency_samples
    in_range  = (eval_idx >= 0) & (eval_idx < len(phase))
    if not in_range.any():
        return None
    edge_idx, eval_idx = edge_idx[in_range], eval_idx[in_range]
    vals = np.angle(np.exp(1j * (phase[eval_idx] - target_rad)), deg=True)
    keep = np.isfinite(vals)
    if not keep.any():
        return None
    return {"time_s": (time_us[edge_idx][keep] - start_ts) / 1e6,
            "values": vals[keep]}


def compute_stimulus_edge_errors(
    trigger_binary: np.ndarray,
    time_us: np.ndarray,
    phase: np.ndarray,
    config: StimulusConfig,
    start_ts: float,
    fs: float,
) -> tuple[dict | None, dict | None]:
    """Phase error at each stimulus trigger edge, measured directly against the
    phase the controller targets.

    For every rising edge of ``trigger_binary`` the error is the wrapped
    difference between ``phase``, read ``erp_latency_s`` seconds after the
    edge, and ``stim_onset_deg`` (see ``_edge_phase_errors`` for why it's a
    read-ahead rather than an analytic correction).

    The offset error is only meaningful in degrees when the burst is itself
    targeted by phase (``stim_dur_unit == "deg"``): there it's measured
    against ``stim_onset_deg + stim_dur_deg``, mirroring
    ``compute_reference_stimulus``, whose reconstructed edges sit at exactly
    that target phase. When ``stim_dur_unit == "ms"`` the burst is timed to a
    fixed duration instead, so a target *phase* for the falling edge doesn't
    exist -- use ``compute_stimulus_duration_error`` for the actual-vs-target
    error in ms there instead, and no offset error is returned here.

    Sign: actual - target, positive == overshoot.

    Returns (onset_error, offset_error) dicts with ``time_s`` / ``values``
    (degrees), or None for an edge kind with nothing to score.
    """
    trigger_binary = np.asarray(trigger_binary)
    time_us        = np.asarray(time_us)
    phase          = np.asarray(phase, dtype=float)

    onset_rad       = np.deg2rad(config.stim_onset_deg)
    latency_samples = int(round(config.erp_latency_s * fs))
    rising          = get_edges(trigger_binary, "rising")
    falling         = get_edges(trigger_binary, "falling")

    onset_err  = _edge_phase_errors(phase, rising, onset_rad, latency_samples,
                                    time_us, start_ts)

    offset_err = None
    if config.stim_dur_unit == "deg":
        dur_rad    = np.deg2rad(config.stim_dur_deg)
        offset_err = _edge_phase_errors(phase, falling, onset_rad + dur_rad,
                                        latency_samples, time_us, start_ts)
    return onset_err, offset_err


def compute_stimulus_duration_error(
    trigger_binary: np.ndarray,
    time_us: np.ndarray,
    config: StimulusConfig,
    start_ts: float,
) -> "dict | None":
    """Actual stimulus burst duration (ms) vs the configured target, measured
    directly off ``trigger_binary`` -- one value per delivered burst, pairing
    each rising edge with the next falling edge after it.

    This is the ``stim_dur_unit == "ms"`` counterpart of the degree-based
    offset error in ``compute_stimulus_edge_errors``: bursts there are timed
    to a fixed millisecond duration rather than a target phase, so the
    natural error to score is actual-minus-target duration in ms rather than
    a phase difference.

    Sign: actual - target, positive == burst ran long.

    Returns a dict with ``time_s`` / ``values`` (ms), or None if there are no
    complete (rising, falling) edge pairs.
    """
    n = min(len(trigger_binary), len(time_us))
    trigger_binary = np.asarray(trigger_binary)[:n]
    time_us        = np.asarray(time_us)[:n]

    rising  = get_edges(trigger_binary, "rising")
    falling = get_edges(trigger_binary, "falling")
    if rising.size == 0 or falling.size == 0:
        return None

    # Pair each rising edge with the next falling edge after it; a trailing
    # rising edge with no matching falling edge (burst cut off at the end of
    # the recording) is dropped.
    match_idx = np.searchsorted(falling, rising, side="right")
    keep      = match_idx < falling.size
    if not keep.any():
        return None
    rising, matched_falling = rising[keep], falling[match_idx[keep]]

    actual_ms = (time_us[matched_falling] - time_us[rising]) / 1000.0
    return {"time_s": (time_us[rising] - start_ts) / 1e6,
            "values": actual_ms - config.stim_dur_ms}
