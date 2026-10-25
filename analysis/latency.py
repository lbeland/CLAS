"""
Pipeline latency analysis from the processors' source timestamps: per-stage
and end-to-end latency, audio onset latency, and UDPSource's sample-rate drift.
compute_latencies() collects them; report.py prints and plot.py plots them.
"""

from dataclasses import dataclass

import numpy as np

from .cache import Recording
from .clas_file_reader import GraphConfig, first_udp_channel_key, parse_graph_structure
from .stimulus import get_edges


@dataclass
class LatencySeries:
    """Latency between two processors, per sample (or per stimulus)."""
    source: str
    target: str
    values: np.ndarray   # latency [us], or [ms] for the audio onset
    ts:     np.ndarray   # source timestamps [us] the values are indexed into, to locate the worst case


@dataclass
class Latencies:
    t0_us:       float                                # first source timestamp, origin of the worst-case times
    stages:      list[LatencySeries]                  # each processor vs its nearest upstream processor [us]
    total:       LatencySeries | None                 # source -> last processor [us]
    audio:       LatencySeries | None                 # StimControl -> recorded trigger onset [ms]
    audio_notes: list[str]                            # why audio is missing, or edge-matching warnings
    udp_true_fs: float | None                         # UDPSource sample rate fitted from its source_ts
    hw_source:   tuple[np.ndarray, np.ndarray] | None # UDPSource (source_ts, hardware_ts) from 0 [us]


def _get_source_ts(proc_name: str, recording: Recording):
    """Resolve source_ts array for a processor by its yaml name."""
    if proc_name in ("UDPSource", "SimulatedSource"):
        return recording.source_ts
    key  = f"{proc_name}_phase" if proc_name == "PhaseEstimation" else proc_name
    data = recording.samples.get(key)
    return data.get("source_ts") if data else None


def compute_latencies(recording: Recording, graph: GraphConfig) -> "Latencies | None":
    """All latencies of one recording, derived from the processors' source
    timestamps. None if fewer than two processors have source_ts."""
    samples = recording.samples
    predecessors, topo_order = parse_graph_structure(graph)

    proc_ts = {p: _get_source_ts(p, recording) for p in topo_order}
    proc_ts = {p: ts for p, ts in proc_ts.items() if ts is not None}
    if len(proc_ts) < 2:
        return None

    def nearest_ancestor(proc: str) -> str | None:
        """Walk up predecessors until a processor with source_ts data is found."""
        current = predecessors.get(proc)
        while current is not None:
            if current in proc_ts:
                return current
            current = predecessors.get(current)
        return None

    # Each processor vs its nearest upstream processor
    stages = []
    for proc in topo_order[1:]:
        ts_proc = proc_ts.get(proc)
        if ts_proc is None:
            continue
        ancestor = nearest_ancestor(proc)
        if ancestor is None:
            continue
        ts_ancestor = proc_ts[ancestor]
        n = min(len(ts_ancestor), len(ts_proc))
        stages.append(LatencySeries(ancestor, proc, ts_proc[:n] - ts_ancestor[:n], ts_ancestor))

    # End-to-end: last processor in topological order vs the source, matched by sample index
    source_proc = next(p for p in topo_order if p in proc_ts)
    last_proc   = next(p for p in reversed(topo_order) if p in proc_ts)
    total = None
    if last_proc != source_proc:
        ts_start, ts_end = proc_ts[source_proc], proc_ts[last_proc]
        n = min(len(ts_start), len(ts_end))
        total = LatencySeries(source_proc, last_proc, ts_end[:n] - ts_start[:n], ts_start)

    audio, audio_notes = None, []
    if "StimControl" in graph.processors and "UDPSource_TRIGGER" in samples:
        audio, audio_notes = _audio_latency(samples["StimControl"], samples["UDPSource_TRIGGER"])

    # hardware_ts (reconstructed ADC sampling time) vs source_ts (UDP receipt
    # time) gap: UDPSource's own acquisition-to-ingestion latency
    udp_true_fs, hw_source = None, None
    udp = samples.get(first_udp_channel_key(samples))
    if udp is not None and udp.get("source_ts") is not None:
        source_ts = udp["source_ts"]
        hw_ts = udp.get("hw_ts")
        n = min(len(hw_ts), len(source_ts)) if hw_ts is not None else len(source_ts)
        if hw_ts is not None:
            hw_source = (source_ts[:n] - np.min(source_ts[:n]), hw_ts[:n] - np.min(hw_ts[:n]))

        # Independent check of fs_eff_: fit source_ts (Falcon's own receipt
        # clock) against sample index, without relying on UDPSource's own RLS
        sample_idx = np.arange(n, dtype=np.float64)
        slope_us_per_sample, _ = np.polyfit(sample_idx, source_ts[:n].astype(np.float64), 1)
        udp_true_fs = 1e6 / slope_us_per_sample

    return Latencies(t0_us=proc_ts[source_proc][0], stages=stages, total=total, audio=audio,
                     audio_notes=audio_notes, udp_true_fs=udp_true_fs, hw_source=hw_source)


def _audio_latency(stim: dict, trig: dict) -> "tuple[LatencySeries | None, list[str]]":
    """Latency [ms] between the computed stimulus onset (StimControl) and the
    recorded trigger onset (UDPSource_TRIGGER), matched edge-by-edge, plus
    notes on why it's missing or how the edges were matched."""
    stim_edges = get_edges((stim["y"] > 0.5).astype(float), "rising")
    trig_edges = get_edges((trig["y"] > 0.5).astype(float), "rising")

    if len(stim_edges) == 0 or len(trig_edges) == 0:
        return None, ["Not enough stimulus/trigger edges for onset latency analysis."]
    if stim.get("source_ts") is None or trig.get("source_ts") is None:
        return None, ["Missing source_ts for stimulus/trigger onset latency analysis."]

    notes = []
    if len(stim_edges) != len(trig_edges):
        notes.append(f"Warning: stimulus/trigger edge count mismatch - "
                     f"stim={len(stim_edges)}, trigger={len(trig_edges)}. Matching by nearest preceding index.")

    # Match each trigger edge to the nearest preceding stim edge (stim_edges
    # is sorted ascending, as returned by get_edges)
    match_pos = np.searchsorted(stim_edges, trig_edges, side="right") - 1
    unmatched = match_pos < 0
    if np.any(unmatched):
        notes.append(f"Warning: {np.sum(unmatched)} trigger edge(s) precede the first "
                     f"stimulus edge and have no valid match - dropping them.")

    trig_edges_matched = trig_edges[~unmatched]
    matched_stim_edges = stim_edges[match_pos[~unmatched]]
    latency_ms = (trig["source_ts"][trig_edges_matched] - stim["source_ts"][matched_stim_edges]) / 1e3
    return LatencySeries("StimControl", "UDPSource_TRIGGER", latency_ms,
                         trig["source_ts"][trig_edges_matched]), notes
