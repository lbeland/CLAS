"""
Phase, frequency and stimulus-edge errors of the online estimates, scored
against ground truth or the offline estimates.
"""

import numpy as np
from scipy.stats import circmean, circstd

from .cache import Recording
from .clas_file_reader import GraphConfig
from .stimulus import StimulusConfig, compute_stimulus_edge_errors, compute_stimulus_duration_error, \
    get_trigger_binary

# LaTeX labels of the phase-error series from compute_errors(). They double as
# plot titles/table headers and as the keys other modules look a series up by,
# so import them from here instead of re-typing the strings. Notation:
# theta = true phase (GT), hat theta = online estimate (ONLINE),
# theta_HT = offline Hilbert phase (HILBERT).
# Phase tracking error, <ESTIMATE>_VS_<REFERENCE> = estimate - reference
ONLINE_VS_GT        = r"$\hat\theta - \theta$"
ONLINE_VS_HILBERT   = r"$\hat\theta - \theta_{\mathrm{HT}}$"
HILBERT_VS_GT       = r"$\theta_{\mathrm{HT}} - \theta$"
# Phase error at the stimulus edges, STIM_<EDGE>_<REFERENCE> = reference phase
# at the trigger edge - target phase
STIM_ONSET_GT        = r"Stim onset ($\theta$)"
STIM_OFFSET_GT       = r"Stim offset ($\theta$)"
STIM_ONSET_HILBERT   = r"Stim onset ($\theta_{\mathrm{HT}}$)"
STIM_OFFSET_HILBERT  = r"Stim offset ($\theta_{\mathrm{HT}}$)"
STIM_ONSET_ONLINE    = r"Stim onset ($\hat\theta$)"
STIM_OFFSET_ONLINE   = r"Stim offset ($\hat\theta$)"
# f0 tracking error of the online estimate vs ground truth / the offline f0_continuous [Hz]
F0_ONLINE_VS_GT      = "Online vs GT f0 error"
F0_ONLINE_VS_HILBERT = "Online vs offline f0 error"


def _edge_trim_window(errors: list[dict], lo_frac: float = 0.1,
                      hi_frac: float = 0.90) -> tuple[float, float]:
    """The [t_lo, t_hi] wall-clock window that drops the leading/trailing
    edge-transient slice of a recording.

    Derived from errors[0]'s timestamps at lo_frac/hi_frac of its length, so
    every series is trimmed by the same amount of time regardless of its own
    sampling rate. Used by plot_errors(), the "phase_error" stack panel, and
    estimate_f0_with_phase(), so the trim shown always matches the trim applied.

    Returns:
        (-inf, +inf) when there is nothing to trim on.
    """
    if not errors:
        return (-np.inf, np.inf)
    ref_t = np.asarray(errors[0]["time_s"], dtype=float)
    if ref_t.size == 0:
        return (-np.inf, np.inf)
    lo_idx = int(lo_frac * len(ref_t))
    hi_idx = min(int(hi_frac * len(ref_t)), len(ref_t) - 1)
    return float(ref_t[lo_idx]), float(ref_t[hi_idx])


def _circ_stats(phi_rad):
    phi = np.asarray(phi_rad, float)
    if phi.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mu  = float(circmean(phi, high=np.pi, low=-np.pi))
    sd  = float(circstd(phi,  high=np.pi, low=-np.pi))
    plv = float(np.abs(np.mean(np.exp(1j * phi))))
    pli = float(np.abs(np.mean(np.sign(np.sin(phi)))))
    return mu, sd, plv, pli


def compute_errors(
    recording: Recording,
    hilbert_phase: np.ndarray,
    f0_continuous: np.ndarray,
    fs: float,
    graph: GraphConfig,
) -> list[dict]:
    """
    Compute phase, f0, and stimulus edge errors, as a list of
    {"label", "time_s", "values", "unit", "linestyle"} series.

    Sign convention for every error series here: estimate - reference
    (wrapped to (-180, 180] for phases). Positive => the estimate leads /
    overshoots the reference.
    """
    errors         = []
    samples        = recording.samples
    start_ts       = recording.time[0]
    time_s         = (recording.time - start_ts) / 1e6  # shared by every processor's output
    true_phase     = recording.ground_truth["true_phase"]
    true_inst_freq = recording.ground_truth["true_inst_freq"]

    # Phase tracking error
    if "PhaseEstimation" in graph.processors:
        online_phase = samples["PhaseEstimation_phase"]["y"]
        if true_phase is not None:
            errors.append({"label": ONLINE_VS_GT, "time_s": time_s, "unit": "degrees", "linestyle": "-",
                           "values": _wrapped_deg(online_phase, true_phase)})
            errors.append({"label": HILBERT_VS_GT, "time_s": time_s, "unit": "degrees", "linestyle": "-",
                           "values": _wrapped_deg(hilbert_phase, true_phase)})
            errors.append({"label": ONLINE_VS_HILBERT, "time_s": time_s, "unit": "degrees", "linestyle": "--",
                           "values": _wrapped_deg(online_phase, hilbert_phase)})
        else:
            errors.append({"label": ONLINE_VS_HILBERT, "time_s": time_s, "unit": "degrees", "linestyle": "-",
                           "values": _wrapped_deg(online_phase, hilbert_phase)})

    # f0 tracking error
    if "FrequencyEstimation" in graph.processors:
        online_f0 = samples["FrequencyEstimation"]["y"]
        if true_inst_freq is not None:
            errors.append({"label": F0_ONLINE_VS_GT, "time_s": time_s, "unit": "Hz",
                           "values": online_f0 - true_inst_freq})
        if f0_continuous is not None:
            errors.append({"label": F0_ONLINE_VS_HILBERT, "time_s": time_s, "unit": "Hz",
                           "values": online_f0 - f0_continuous})

    # Stimulus edge errors
    if "StimControl" in graph.processors:
        trigger_binary = get_trigger_binary(samples)
        stim_cfg       = StimulusConfig.from_graph(graph)

        if true_phase is not None:
            ref_phase = true_phase
            onset_label, offset_label = STIM_ONSET_GT, STIM_OFFSET_GT
        else:
            ref_phase = hilbert_phase
            onset_label, offset_label = STIM_ONSET_HILBERT, STIM_OFFSET_HILBERT

        def add_edge_errors(phase, onset_label, offset_label, linestyle):
            """Phase at each trigger onset/offset edge vs the controller's target phase."""
            onset_err, offset_err = compute_stimulus_edge_errors(trigger_binary, recording.time, phase,
                                                                 stim_cfg, start_ts, fs)
            if onset_err is not None:
                errors.append({"label": onset_label, "time_s": onset_err["time_s"],
                               "values": onset_err["values"], "unit": "degrees",
                               "linestyle": linestyle})
            if offset_err is not None:
                errors.append({"label": offset_label, "time_s": offset_err["time_s"],
                               "values": offset_err["values"], "unit": "degrees",
                               "linestyle": linestyle})

        # Scored against true_phase if available, else the offline Hilbert phase
        add_edge_errors(ref_phase, onset_label, offset_label, "-")

        # Same edges, scored against the online PhaseEstimation output instead
        if "PhaseEstimation" in graph.processors:
            add_edge_errors(samples["PhaseEstimation_phase"]["y"], STIM_ONSET_ONLINE, STIM_OFFSET_ONLINE,
                            ":")

        # "ms" mode times the burst rather than phase-targeting the falling edge,
        # so score actual-vs-target burst length in ms instead
        if stim_cfg.stim_dur_unit == "ms":
            dur_err = compute_stimulus_duration_error(trigger_binary, recording.time, stim_cfg, start_ts)
            if dur_err is not None:
                errors.append({"label": "Stim duration error",
                               "time_s": dur_err["time_s"], "values": dur_err["values"],
                               "unit": "ms"})

    return errors


def _wrapped_deg(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """estimate - reference [rad], wrapped to (-180, 180] degrees."""
    return np.angle(np.exp(1j * (estimate - reference)), deg=True)
