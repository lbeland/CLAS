"""
Stimulus reconstruction and edge-error analysis.

StimulusConfig is the single place to add new targeting rules - add fields here
and update from_graph() and compute_reference_stimulus() accordingly.
"""

from dataclasses import dataclass
import numpy as np

from .cache import Recording
from .clas_file_reader import GraphConfig

# Study conditions: StimControl's stim_onset_deg, and its display name
STIM_ONSET_DEGS = (0, 60, 150, 180, 240, 330)

CONDITION_LABELS = {
    330: "Pre-Peak",
    180: r"$\mathrm{P}_1$-Trough",
    60:  "Post-Peak",
    150: "Pre-Trough",
    0:   r"$\mathrm{P}_1$-Peak",
    240: "Post-Trough",
}


@dataclass
class StimulusConfig:
    stim_onset_deg: float = 0.0
    stim_dur_deg:   float = 90.0
    stim_dur_unit:  str   = "deg"
    stim_dur_ms:    float = 20.0
    p1_latency_ms:  float = 0.0

    @classmethod
    def from_graph(cls, graph: GraphConfig) -> "StimulusConfig":
        opts = graph.options("StimControl")
        return cls(
            stim_onset_deg=opts.get("stim_onset_deg", 0.0),
            stim_dur_deg=opts.get("stim_dur_deg",   90.0),
            stim_dur_unit=opts.get("stim_dur_unit", "deg"),
            stim_dur_ms=opts.get("stim_dur_ms",     20.0),
            # older recordings store the option as erp_latency_ms / erp_latency_s
            p1_latency_ms=opts.get(
                "p1_latency_ms",
                opts.get("erp_latency_ms", opts.get("erp_latency_s", 0.0) * 1000.0),
            ),
        )


def get_edges(signal: np.ndarray, kind: str) -> np.ndarray:
    if kind == "rising":
        return np.where((signal[1:] == 1) & (signal[:-1] == 0))[0] + 1
    return np.where((signal[1:] == 0) & (signal[:-1] == 1))[0] + 1


def get_trigger_binary(samples: dict) -> "np.ndarray | None":
    """The delivered stimulus as 0/1: the hardware trigger (StimTrak, on the
    UDPSource aux port) if recorded, else StimControl's software trigger.
    None if neither is there."""
    if "UDPSource_TRIGGER" in samples:
        trigger_y = samples["UDPSource_TRIGGER"]["y"]
    elif "StimControl" in samples:
        trigger_y = samples["StimControl"]["y"]
    else:
        return None
    return (trigger_y > 0.5).astype(float)


def compute_target_stimulus(recording: Recording, hilbert_phase: np.ndarray, f0_continuous: np.ndarray,
                            fs: float, graph: GraphConfig) -> np.ndarray:
    """The ideal stimulus train StimControl should have produced,
    reconstructed from true_phase if available, else the
    offline Hilbert phase. For visualisation / EDF export."""
    ground_truth = recording.ground_truth
    if ground_truth["true_phase"] is not None:
        ref_phase = ground_truth["true_phase"]
    else:
        ref_phase = hilbert_phase

    if ground_truth["true_inst_freq"] is not None:
        f0 = ground_truth["true_inst_freq"]
    else:
        f0 = f0_continuous

    return compute_reference_stimulus(ref_phase, f0, fs, StimulusConfig.from_graph(graph))


def compute_reference_stimulus(
    phase: np.ndarray,
    f0: np.ndarray,
    fs: float,
    config: StimulusConfig,
) -> np.ndarray:
    """Reconstruct the ideal stimulus signal from a reference phase time
    series, mirroring StimControl::Process().

    Args:
        phase: Reference phase series (rad).
        f0: Fundamental frequency series (Hz), same length as ``phase``.
        fs: Sample rate (Hz).
        config: Stimulus targeting parameters.

    Returns:
        Binary ndarray, 1 inside each stimulus window.
    """
    f0_safe   = np.isfinite(f0) & (f0 > 0)
    onset_rad = np.deg2rad(config.stim_onset_deg)
    p1_latency_s = config.p1_latency_ms / 1000.0

    # Correct target phase by P1 latency
    phase_past_onset = np.mod(phase + 2.0 * np.pi * f0 * p1_latency_s - onset_rad, 2.0 * np.pi)

    # Onset candidates: within the first 10% of a cycle past the target phase
    candidates = np.where(f0_safe & (phase_past_onset < 0.1 * 2.0 * np.pi))[0]

    # Accept a candidate only 1 cycle + 10% buffer after the last onset
    onsets, last_onset = [], 0
    for i in candidates:
        if (i - last_onset) / fs >= 1 / (f0[i] + f0[i] * 0.1):
            onsets.append(i)
            last_onset = i

    # Fixed duration per onset instead of phase-targeted offset
    stim_ref = np.zeros_like(phase, dtype=float)
    for i in onsets:
        dur_s = config.stim_dur_deg / 360.0 / f0[i] if config.stim_dur_unit == "deg" else config.stim_dur_ms / 1000.0
        stim_ref[i : i + int(round(dur_s * fs))] = 1
    return stim_ref


def _edge_phase_errors(
    phase: np.ndarray,
    edge_idx: np.ndarray,
    target_rad: float,
    latency_samples: int,
    time_us: np.ndarray,
    start_ts: float,
) -> "dict | None":
    """Wrapped phase error (degrees, actual - target) at each sample in
    ``edge_idx``, offset by ``latency_samples``.

    Args:
        phase: Reference phase series (rad).
        edge_idx: Sample indices to evaluate.
        target_rad: Target phase (rad).
        latency_samples: Offset applied to each edge index before reading ``phase``.
        time_us: Timestamps matching ``phase``, in microseconds.
        start_ts: Recording start time, same units as ``time_us``.

    Returns:
        {"time_s", "values"} in degrees, or None if no edge survives (out of
        range, or a non-finite ``phase`` value).
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
    """Phase error (degrees, actual - target) at each stimulus trigger edge,
    measured against the phase the controller targets.

    Onset error is the rising-edge phase vs. ``stim_onset_deg``. Offset error
    is the falling-edge phase vs. ``stim_onset_deg + stim_dur_deg``, only
    returned when ``config.stim_dur_unit == "deg"`` (use
    ``compute_stimulus_duration_error`` for the "ms" case instead).

    Returns:
        (onset_error, offset_error): {"time_s", "values"} dicts
    """
    trigger_binary = np.asarray(trigger_binary)
    time_us        = np.asarray(time_us)
    phase          = np.asarray(phase, dtype=float)

    onset_rad       = np.deg2rad(config.stim_onset_deg)
    latency_samples = int(round(config.p1_latency_ms / 1000.0 * fs))
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
    """Actual stimulus burst duration (ms, actual - target) vs the configured
    target, one value per delivered burst (paired rising/falling edges of
    ``trigger_binary``). The ``stim_dur_unit == "ms"`` counterpart of
    ``compute_stimulus_edge_errors``'s degree-based offset error.

    Returns:
        {"time_s", "values"} in ms, or None if there are no complete edge pairs.
    """
    n = min(len(trigger_binary), len(time_us))
    trigger_binary = np.asarray(trigger_binary)[:n]
    time_us        = np.asarray(time_us)[:n]

    rising  = get_edges(trigger_binary, "rising")
    falling = get_edges(trigger_binary, "falling")
    if rising.size == 0 or falling.size == 0:
        return None

    # Pair each rising edge with the next falling edge; drop a trailing rising
    # edge with no match (burst cut off at the end of the recording)
    match_idx = np.searchsorted(falling, rising, side="right")
    keep      = match_idx < falling.size
    if not keep.any():
        return None
    rising, matched_falling = rising[keep], falling[match_idx[keep]]

    actual_ms = (time_us[matched_falling] - time_us[rising]) / 1000.0
    return {"time_s": (time_us[rising] - start_ts) / 1e6,
            "values": actual_ms - config.stim_dur_ms}
