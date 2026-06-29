"""
Data loading: file discovery, processor signal loading, ground-truth extraction,
graph structure parsing, and latency reporting.
"""

import os
import numpy as np
from pathlib import Path
from collections import deque

from .read_output import get_signal_data

RESULTS_DIR = "_last_run"


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


def _trim_falling_edges(y: np.ndarray, fs: float, trim_ms: float) -> np.ndarray:
    """Shift each falling edge earlier by trim_ms ms; preserves pulse onsets exactly."""
    trim_samples = int(round(trim_ms * fs / 1000))
    y = y.copy()
    falling = np.where((y[:-1] == 1) & (y[1:] == 0))[0]
    raising  = np.where((y[:-1] == 0) & (y[1:] == 1))[0]
    for idx, f in enumerate(falling):
        if idx >= len(raising):
            continue
        start = max(raising[idx], f - trim_samples + 1)
        y[start : f + 1] = 0.0
    return y


def load_processor_signals(fs, results_dir, processors: list[str], timestamps: bool = True) -> dict:
    """Load raw signal data from all processors into a dict keyed by label."""
    samples = {}

    for processor in processors:
        if processor == "Producer":
            file = get_results_file(processor, slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(10)), timestamps=timestamps)
            if signal is None:
                continue
            source_time = time["source_ts"]
            time = time["hardware_ts"]
            for idx, channel in enumerate(signal.T):
                samples[f"{processor}_{idx}"] = {"x": time, "y": channel, "source_ts": source_time}

            file = get_results_file(processor, name=".meta_out", slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(3)), timestamps=timestamps)
            if signal is None:
                continue
            time = time["hardware_ts"]
            for idx, channel in enumerate(signal.T):
                samples[f"{processor}_meta_{idx}"] = {"x": time, "y": channel}

        elif processor == "SourceClient":
            file = get_results_file(processor, slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(32)), timestamps=timestamps)
            if signal is None:
                continue
            source_time = time["source_ts"]
            time = time["hardware_ts"]
            for idx, channel in enumerate(signal.T):
                samples[f"SourceClient_{idx}"] = {"x": time, "y": channel, "source_ts": source_time}

            file = get_results_file(processor, slot=1, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(8)), timestamps=timestamps)
            if signal is not None:
                time = time["hardware_ts"]
                samples["SourceClient_AUX"] = {"x": time, "y": signal[:, 0] if signal.ndim > 1 else signal}

            file = get_results_file(processor, slot=1, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=8, timestamps=timestamps)
            if signal is not None:
                time = time["hardware_ts"]
                binary = (signal > 0.5).astype(float)
                samples["SourceClient_TRIGGER"] = {"x": time, "y": _trim_falling_edges(binary, fs, 11.0)}

        elif processor == "PhaseEstimator":
            file = get_results_file(processor, slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
            if signal is not None:
                samples[f"{processor}_phase"] = {"x": time["hardware_ts"], "y": signal, "source_ts": time["source_ts"]}

            file = get_results_file(processor, slot=1, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
            if signal is not None:
                samples[f"{processor}_real"] = {"x": time["hardware_ts"], "y": signal, "source_ts": time["source_ts"]}

        elif processor == "StimulusController":
            file = get_results_file(processor, slot=0, results_dir=results_dir)
            if file is not None:
                signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
                if signal is not None:
                    samples[processor] = {"x": time["hardware_ts"], "y": signal, "source_ts": time["source_ts"]}

        else:
            result_name = "ch_idx_out" if processor == "ChannelSelector" else ".out"
            file = get_results_file(processor, name=result_name, slot=0, results_dir=results_dir)
            if file is not None:
                signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
                if signal is not None:
                    samples[processor] = {"x": time["hardware_ts"], "y": signal, "source_ts": time["source_ts"]}

    first_timestamps = [samples[key]["x"][0] for key in samples]
    assert len(set(first_timestamps)) == 1 or len(set(first_timestamps)) == 0, \
        "Mismatched timestamps across processors"

    min_lengths = min(len(samples[key]["x"]) for key in samples)
    for key in samples:
        samples[key]["x"] = samples[key]["x"][:min_lengths]
        samples[key]["y"] = samples[key]["y"][:min_lengths]
        if samples[key].get("source_ts") is not None:
            samples[key]["source_ts"] = samples[key]["source_ts"][:min_lengths]

    return samples


def extract_ground_truth(samples: dict) -> dict | None:
    """Extract ground-truth signals from SourceClient or Producer output."""
    if "SourceClient_0" in samples:
        if "ChannelSelector" in samples:
            channel_selected = samples["ChannelSelector"]["y"] - 1  # convert to zero-based
            raw = []
            for sample_idx, select_idx in enumerate(channel_selected):
                raw.append(samples[f"SourceClient_{select_idx}"]["y"][sample_idx])
            raw = np.array(raw)
            time        = samples[f"SourceClient_{select_idx}"]["x"]
            source_time = samples[f"SourceClient_{select_idx}"]["source_ts"]
        else:
            raw         = samples["SourceClient_0"]["y"]
            time        = samples["SourceClient_0"]["x"]
            source_time = samples["SourceClient_0"]["source_ts"]

        if os.path.exists("simulated_signal.npy"):
            loaded = np.load("simulated_signal.npy")
            assert raw[0, 0] == loaded["value"][0], \
                "Loaded simulated signal does not match SourceClient signal"
            return {
                "raw":            raw,
                "time":           time,
                "true_amplitude": loaded["amplitude"],
                "true_phase":     np.angle(np.exp(1j * loaded["phase"])),
                "true_inst_freq": loaded["inst_freq"],
                "source_ts":      loaded["source_ts"],
            }
        return {
            "raw": raw, "time": time, "source_ts": source_time,
            "true_amplitude": None, "true_phase": None, "true_inst_freq": None,
        }

    elif "Producer_0" in samples:
        if "ChannelSelector" in samples:
            channel_selected = samples["ChannelSelector"]["y"]
            raw = []
            for sample_idx, select_idx in enumerate(channel_selected):
                raw.append(samples[f"Producer_{select_idx-1}"]["y"][sample_idx])
            raw         = np.array(raw)
            time        = samples[f"Producer_{select_idx-1}"]["x"]
            source_time = samples[f"Producer_{select_idx-1}"]["source_ts"]
        else:
            raw         = samples["Producer_0"]["y"]
            time        = samples["Producer_0"]["x"]
            source_time = samples["Producer_0"]["source_ts"]

        return {
            "raw":            raw,
            "time":           time,
            "source_ts":      source_time,
            "true_amplitude": samples.get("Producer_meta_0", {}).get("y"),
            "true_phase":     np.angle(np.exp(1j * samples["Producer_meta_1"]["y"])),
            "true_inst_freq": samples.get("Producer_meta_2", {}).get("y"),
        }

    return None


# ---------------------------------------------------------------------------
# Graph structure + latency
# ---------------------------------------------------------------------------

def _parse_graph_structure(graph_config: dict) -> tuple[dict, list]:
    """Parse graph connections → (predecessors dict, topological order).
    Serializers and Consumers are excluded."""
    processors  = graph_config.get("graph", {}).get("processors", {})
    connections = graph_config.get("graph", {}).get("connections", [])

    skip_classes = {"FileSerializer", "Consumer"}
    skip_names = {name for name, cfg in processors.items()
                  if isinstance(cfg, dict) and cfg.get("class") in skip_classes}
    skip_names |= {name for name in processors if "Serializer" in name or "Consumer" in name}

    all_procs = [name for name in processors if name not in skip_names]
    edges      = {p: [] for p in all_procs}
    predecessors: dict[str, str] = {}
    in_degree  = {p: 0 for p in all_procs}

    for conn in connections:
        if not isinstance(conn, str):
            continue
        parts = conn.split("=")
        if len(parts) != 2:
            continue
        from_proc = parts[0].strip().split(".")[0]
        to_proc   = parts[1].strip().split(".")[0]
        if from_proc in skip_names or to_proc in skip_names:
            continue
        if from_proc in all_procs and to_proc in all_procs:
            if to_proc not in predecessors:
                predecessors[to_proc] = from_proc
            edges[from_proc].append(to_proc)
            in_degree[to_proc] += 1

    queue = deque(sorted(p for p in all_procs if in_degree[p] == 0))
    order = []
    while queue:
        proc = queue.popleft()
        order.append(proc)
        for nxt in sorted(edges[proc]):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    return predecessors, order


def _get_source_ts(proc_name: str, samples: dict, ground_truth: dict):
    """Resolve source_ts array for a processor by its yaml name."""
    if proc_name in ("SourceClient", "Producer"):
        return ground_truth.get("source_ts")
    key  = f"{proc_name}_phase" if proc_name == "PhaseEstimator" else proc_name
    data = samples.get(key)
    return data.get("source_ts") if data else None


def print_latencies(samples: dict, ground_truth: dict, graph_config: dict) -> None:
    """Print a per-stage latency table derived from source_timestamps."""
    predecessors, topo_order = _parse_graph_structure(graph_config)

    proc_ts = {p: _get_source_ts(p, samples, ground_truth) for p in topo_order}
    proc_ts = {p: ts for p, ts in proc_ts.items() if ts is not None}

    if len(proc_ts) < 2:
        print("Not enough processors with source_ts for latency analysis.")
        return

    def nearest_ancestor(proc: str) -> str | None:
        """Walk up predecessors until a processor with source_ts data is found."""
        current = predecessors.get(proc)
        while current is not None:
            if current in proc_ts:
                return current
            current = predecessors.get(current)
        return None

    col = 22
    W   = col * 2 + 52
    print(f"\n{'─' * W}")
    print(f"  Pipeline latency analysis")
    print(f"{'─' * W}")
    print(f"  {'From':<{col}} {'To':<{col}} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8} {'Idx':>8}  ms")
    print(f"{'─' * W}")

    source_proc = next((p for p in topo_order if p in proc_ts), None)

    for proc in topo_order[1:]:
        ts_proc = proc_ts.get(proc)
        if ts_proc is None:
            continue
        ancestor = nearest_ancestor(proc)
        if ancestor is None:
            continue
        ts_ancestor = proc_ts[ancestor]
        n   = min(len(ts_ancestor), len(ts_proc))
        lat = (ts_proc[:n] - ts_ancestor[:n]) / 1e3   # µs → ms
        print(f"  {ancestor:<{col}} {proc:<{col}} "
              f"{np.mean(lat):>8.2f} {np.median(lat):>8.2f} "
              f"{np.percentile(lat, 95):>8.2f} {np.max(lat):>8.2f} {np.argmax(lat):>8}")

    last_proc = next((p for p in reversed(topo_order) if p in proc_ts), None)
    if source_proc and last_proc and last_proc != source_proc:
        ts_start = proc_ts[source_proc]
        ts_end   = proc_ts[last_proc]
        n     = min(len(ts_start), len(ts_end))
        total = (ts_end[:n] - ts_start[:n]) / 1e3
        print(f"{'─' * W}")
        print(f"  {'TOTAL  ' + source_proc:<{col}} {last_proc:<{col}} "
              f"{np.mean(total):>8.2f} {np.median(total):>8.2f} "
              f"{np.percentile(total, 95):>8.2f} {np.max(total):>8.2f} {np.argmax(total):>8}")
    print(f"{'─' * W}\n")
