"""
Readers for the files a CLAS session writes into its results folder: the
processors' serialized .bin output, the stim protocol log, and the graph
config (.yaml).
"""

import csv
import datetime
import glob
import os
import re
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import yaml

RESULTS_DIR = "_last_run"

CHANNEL_NAMES = {
    1: "Fp1",
    2: "Fpz",
    3: "Fz",
    4: "F3",
    5: "F7",
    27: "F4",
    29: "Fp2",
    28: "F8",
    9: "C3",
    25: "C4",
    6: "E1",
    30: "E2",
    10: "Cz",
    26: "FCz",
    12: "M1",
    23: "M2",
}

# E1/E2 are EOG (eye) channels, not scalp positions -- excluded from CSD/
# surface-Laplacian computations even though they're in CHANNEL_NAMES.
EOG_CHANNEL_NAMES = ("E1", "E2")


# ---------------------------------------------------------------------------
# Results folder discovery
# ---------------------------------------------------------------------------

def find_session_run_dirs(base_dir: str = "results", session_glob: str = "CLAS_*",
                          graph_name: str = "TurboLinkCLAS.yaml") -> list[str]:
    """Every graph_name run directory nested one level under a session folder
    matching session_glob (e.g. results/CLAS_linda/<run>)."""
    session_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, session_glob)) if os.path.isdir(d))
    return sorted({
        os.path.dirname(p)
        for session_dir in session_dirs
        for p in glob.glob(os.path.join(session_dir, "*", graph_name))
    })


def find_result_dirs(base_dir: str = "results") -> list[str]:
    """Every directory under base_dir that directly contains a *.yaml graph
    config, i.e. every valid analyse_results() target."""
    return sorted({
        os.path.dirname(p) for p in glob.glob(os.path.join(base_dir, "**", "*.yaml"), recursive=True)
    })


# ---------------------------------------------------------------------------
# Graph config
# ---------------------------------------------------------------------------

@dataclass
class GraphConfig:
    """The parts of a graph .yaml the analysis uses."""
    path:        str
    raw:         dict          # full parsed yaml, for anything not covered below
    fs:          float | None  # graph.defaults.fs
    processors:  dict          # name -> processor config
    connections: list

    def options(self, processor: str) -> dict:
        """The options of one processor, or {} if it isn't in the graph."""
        return (self.processors.get(processor) or {}).get("options") or {}


def load_graph_config(results_dir: str, pattern: str = "*.yaml") -> "GraphConfig | None":
    """The graph .yaml in results_dir matching pattern, or None if there is none.

    Raises:
        ValueError: if the graph has no processors.
    """
    graph_files = glob.glob(os.path.join(results_dir, pattern))
    if not graph_files:
        return None
    with open(graph_files[0]) as f:
        raw = yaml.safe_load(f)

    graph = raw.get("graph", {})
    if not graph.get("processors"):
        raise ValueError(f"No processors found in {graph_files[0]}.")
    return GraphConfig(
        path=graph_files[0], raw=raw,
        fs=graph.get("defaults", {}).get("fs"),
        processors=graph["processors"],
        connections=graph.get("connections") or [],
    )


def parse_graph_structure(graph: GraphConfig) -> tuple[dict, list]:
    """Parse graph connections into (predecessors dict, topological order).
    Serializers and BenchSinks are excluded."""
    processors  = graph.processors
    connections = graph.connections

    skip_classes = {"FileSerializer", "BenchSink"}
    skip_names = {name for name, cfg in processors.items()
                  if isinstance(cfg, dict) and cfg.get("class") in skip_classes}
    skip_names |= {name for name in processors if "Serializer" in name or "BenchSink" in name}

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


# ---------------------------------------------------------------------------
# Stim protocol log
# ---------------------------------------------------------------------------

def load_stim_annotations(results_dir: str, start_ts: float) -> "mne.Annotations | None":
    """Load stim event markers from a stim protocol log CSV in results_dir, if present.

    Each "on" or "off" row opens a segment that runs until the next logged row
    (whichever state that is), producing a "stim_on"/"stim_off" duration
    annotation. Non on/off rows (protocol_start/end/stopped) additionally get
    their own zero-duration marker.

    start_ts is the recording start time in the same units as recording.time
    (microseconds since epoch), used to align log timestamps to the EDF timeline.
    """
    log_files = sorted(glob.glob(os.path.join(results_dir, "stim_protocol_log_*.csv")))
    if not log_files:
        return None

    with open(log_files[-1], newline="") as f:
        reader = csv.DictReader(f)
        rows = [(datetime.datetime.fromisoformat(row["timestamp"]).timestamp() - start_ts / 1e6,
                 row["state"]) for row in reader]

    onsets, durations, descriptions = [], [], []
    prev_time, prev_state = None, None
    for t, state in rows:
        if prev_state in ("on", "off"):
            onsets.append(prev_time)
            durations.append(t - prev_time)
            descriptions.append("stim_on" if prev_state == "on" else "stim_off")
        if state not in ("on", "off"):
            onsets.append(t)
            durations.append(0.0)
            descriptions.append(state)
        prev_time, prev_state = t, state

    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


# ---------------------------------------------------------------------------
# .bin file parsing
# ---------------------------------------------------------------------------

TYPE_FORMAT = {
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "float32": "f",
    "float64": "d",
}

TYPE_NUMPY = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
}

def parse_data_entry(entry):
    # Expected format: "name dtype (d0,d1,...)" or "name dtype (d0)"
    match = re.match(r"^(.+?)\s+(\w+)\s*\(([^)]*)\)\s*$", entry)
    if not match:
        raise ValueError(f"Unsupported data description entry: {entry}")

    name = match.group(1).strip()
    dtype = match.group(2).strip().lower()
    dims = [int(part.strip()) for part in match.group(3).split(",") if part.strip()]
    if not dims:
        raise ValueError(f"Missing dimensions in entry: {entry}")

    return name, dtype, dims

def item_count(dims):
    count = 1
    for d in dims:
        count *= d
    return count

def infer_record_layout(header_data):
    if not isinstance(header_data, list) or not header_data:
        raise ValueError("Header does not contain a valid 'data' description list")

    layout = []
    record_size = 0

    for entry in header_data:
        name, dtype, dims = parse_data_entry(entry)
        fmt = TYPE_FORMAT.get(dtype)
        if fmt is None:
            raise ValueError(f"Unsupported dtype in header entry: {dtype}")

        n_items = item_count(dims)
        byte_size = struct.calcsize("<" + fmt) * n_items
        layout.append(
            {
                "name": name,
                "dtype": dtype,
                "dims": dims,
                "n_items": n_items,
                "byte_size": byte_size,
                "offset": record_size,
            }
        )
        record_size += byte_size

    return layout, record_size


def _find_header_end(blob: bytes) -> int:
    # Binary output files start with a YAML header terminated by a document end marker.
    # Support both LF and CRLF line endings.
    for marker in (b"...\n", b"...\r\n"):
        idx = blob.find(marker)
        if idx != -1:
            return idx + len(marker)
    raise ValueError("Could not locate YAML header terminator ('...') in file")


def _extract_field(payload: bytes, layout: list, record_size: int, n_records: int, name: str):
    meta = next((field for field in layout if field["name"] == name), None)
    if meta is None:
        return None

    dtype = TYPE_NUMPY[meta["dtype"]]
    offset = meta["offset"]
    n_items = meta["n_items"]

    flat = np.ndarray(
        shape=(n_records, n_items),
        dtype=dtype,
        buffer=payload,
        offset=offset,
        strides=(record_size, np.dtype(dtype).itemsize),
    )

    # Reshape back to declared dimensions
    dims = meta["dims"]
    if dims == [1]:
        return flat[:, 0]
    return flat.reshape((n_records, *dims))

def get_signal_data(path, channel=0, timestamps=False):
    if path is None:
        print("No path provided for signal data")
        return None, None
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None, None
    with open(path, "rb") as f:
        blob = f.read()

    header_end = _find_header_end(blob)
    header = yaml.safe_load(blob[:header_end])
    payload = blob[header_end:]

    # print("Header:", header)

    layout, record_size = infer_record_layout(header.get("data"))

    n_records = len(payload) // record_size

    if n_records == 0:
        return None, None

    payload = payload[: n_records * record_size]

    signal_meta = next((field for field in layout if field["name"] == "signal" or field["name"] == "scalar_data"), None)
    if signal_meta is None:
        raise ValueError("No 'signal' or 'scalar_data' field found in header data description")

    signal_dtype = TYPE_NUMPY[signal_meta["dtype"]]
    signal_offset = signal_meta["offset"]
    signal_n_items = signal_meta["n_items"]
    signal_flat = np.ndarray(
        shape=(n_records, signal_n_items),
        dtype=signal_dtype,
        buffer=payload,
        offset=signal_offset,
        strides=(record_size, np.dtype(signal_dtype).itemsize),
    )

    samples = signal_flat.reshape((n_records, *signal_meta["dims"]))
    if len(signal_meta["dims"]) == 2:
        # channel=-1 keeps every channel; the channel count is taken from the
        # header's declared dimensions rather than being passed in.
        if channel != -1:
            samples = samples[:, channel, :]
    print(os.path.basename(path), "samples shape:", samples.shape, "dtype:", samples.dtype)

    if timestamps:
        source_ts = _extract_field(payload, layout, record_size, n_records, "source_ts")
        hardware_ts = _extract_field(payload, layout, record_size, n_records, "hardware_ts")
        # Signed so downstream subtractions (latency diffs) don't wrap around on
        # negative results; microsecond epoch counts fit int64 comfortably.
        if source_ts is not None:
            source_ts = source_ts.astype(np.int64)
        if hardware_ts is not None:
            hardware_ts = hardware_ts.astype(np.int64)
        # Keep the return value extensible: callers can pull what they need.
        ts = {
            "source_ts": source_ts,
            "hardware_ts": hardware_ts,
        }
        return samples.squeeze(), ts
    else:
        return samples, None


# ---------------------------------------------------------------------------
# Processor signals
# ---------------------------------------------------------------------------

def first_udp_channel_key(keys) -> str | None:
    """The lowest-indexed "UDPSource_{idx}" raw channel key among `keys`, or
    None if there isn't one. Not all channels are necessarily recorded (e.g.
    only channel 3 selected), so callers that need "the" UDPSource channel
    (the hardware->source latency calc, and the one true hw_ts array kept in
    runtime_metadata.h5) can't just assume index 0 exists."""
    indices = sorted(
        int(k.split("_")[1]) for k in keys
        if k.startswith("UDPSource_") and k.split("_")[1].isdigit()
    )
    return f"UDPSource_{indices[0]}" if indices else None


def _iter_channels(signal: np.ndarray):
    """Yield (n_samples,) columns from `signal`, whether it's 1D (single channel,
    as returned by get_signal_data's squeeze) or 2D (n_samples, n_channels)."""
    signal = np.asarray(signal)
    if signal.ndim == 1:
        return [signal]
    return signal.T


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
        if processor == "SimulatedSource":
            file = get_results_file(processor, name=".data", slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
            if signal is None:
                continue
            source_time = time["source_ts"]
            time = time["hardware_ts"]
            for idx, channel in enumerate(_iter_channels(signal)):
                samples[f"{processor}_{idx}"] = {"x": time, "y": channel, "source_ts": source_time}

            file = get_results_file(processor, name=".meta", slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=list(range(3)), timestamps=timestamps)
            if signal is None:
                continue
            time = time["hardware_ts"]
            for idx, channel in enumerate(_iter_channels(signal)):
                samples[f"{processor}_meta_{idx}"] = {"x": time, "y": channel}

        elif processor == "UDPSource":
            file = get_results_file(processor, name=".eeg", slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=-1, timestamps=timestamps)
            if signal is None:
                continue
            source_time = time["source_ts"]
            time = time["hardware_ts"]
            for idx, channel in enumerate(_iter_channels(signal)):
                samples[f"UDPSource_{idx}"] = {"x": time, "y": channel, "source_ts": source_time}

            file = get_results_file(processor, name=".aux", slot=0, results_dir=results_dir)
            # AUX channels are stored on the aux port, channels 0-7
            signal, time = get_signal_data(file, channel=list(range(8)), timestamps=timestamps)
            if signal is not None:
                source_time = time["source_ts"]
                time = time["hardware_ts"]
                for idx, channel in enumerate(_iter_channels(signal)):
                    samples[f"UDPSource_AUX_{idx}"] = {"x": time, "y": channel, "source_ts": source_time}
            # Trigger is stored on the aux port, channel 8
            signal, time = get_signal_data(file, channel=8, timestamps=timestamps)
            if signal is not None:
                source_time = time["source_ts"]
                time = time["hardware_ts"]
                binary = (signal > 0.5).astype(float)
                samples["UDPSource_TRIGGER"] = {"x": time, "y": _trim_falling_edges(binary, fs, 11.0), "source_ts": source_time}

        elif processor == "PhaseEstimation":
            file = get_results_file(processor, name=".phase", slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
            if signal is not None:
                samples[f"{processor}_phase"] = {"x": time["hardware_ts"], "y": signal, "source_ts": time["source_ts"]}

            file = get_results_file(processor, name=".real", slot=0, results_dir=results_dir)
            signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
            if signal is not None:
                samples[f"{processor}_real"] = {"x": time["hardware_ts"], "y": signal, "source_ts": time["source_ts"]}
        elif processor == "FrequencyEstimation":
            # slot 0: Kalman-smoothed f0 estimate (always present)
            file = get_results_file(processor, name=".out", slot=0, results_dir=results_dir)
            if file is not None:
                signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
                if signal is not None:
                    samples[processor] = {"x": time["hardware_ts"], "y": signal, "source_ts": time["source_ts"]}

            # slot 1: raw peak f0, pre-Kalman (only when debug_output=true and out.1 is serialized)
            file = get_results_file(processor, name=".out", slot=1, results_dir=results_dir)
            if file is not None:
                signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
                if signal is not None:
                    samples[f"{processor}_raw"] = {"x": time["hardware_ts"], "y": signal, "source_ts": time["source_ts"]}
        else:
            file = get_results_file(processor, name=".out", slot=0, results_dir=results_dir)
            if file is not None:
                signal, time = get_signal_data(file, channel=0, timestamps=timestamps)
                if signal is not None:
                    samples[processor] = {"x": time["hardware_ts"], "y": signal, "source_ts": time["source_ts"]}

    first_timestamps = [samples[key]["x"][0] for key in samples]
    assert len(set(first_timestamps)) == 1 or len(set(first_timestamps)) == 0, \
        "Mismatched timestamps across processors"

    if len(samples) == 0:
        print("No processor signals found in results directory.")
        return {}

    min_lengths = min(len(samples[key]["x"]) for key in samples)
    start_ts = first_timestamps[0]
    nominal_x = start_ts + np.round(np.arange(min_lengths) / fs * 1e6).astype(np.int64)
    for key in samples:
        # "x" becomes the idealized uniform grid; only UDPSource keeps its
        # reconstructed ADC times, under "hw_ts", for its own latency calc
        if key.startswith("UDPSource"):
            samples[key]["hw_ts"] = samples[key]["x"][:min_lengths]
        samples[key]["x"] = nominal_x
        samples[key]["y"] = samples[key]["y"][:min_lengths]
        if samples[key].get("source_ts") is not None:
            samples[key]["source_ts"] = samples[key]["source_ts"][:min_lengths]

    return samples


def extract_selected_signal(samples: dict, graph: GraphConfig) -> dict | None:
    """{"raw", "time", "source_ts"} of the signal the pipeline processed: per
    sample, the UDPSource/SimulatedSource channel ChannelSelection picked (or
    channel 0 without a ChannelSelection). None if nothing was loaded or the
    graph has no source."""
    if not samples:
        return None
    if "UDPSource" in graph.processors:
        prefix = "UDPSource"
    elif "SimulatedSource" in graph.processors:
        prefix = "SimulatedSource"
    else:
        return None

    if "ChannelSelection" in graph.processors:
        channel_selected = samples["ChannelSelection"]["y"] - 1  # convert to zero-based
        raw = []
        for sample_idx, select_idx in enumerate(channel_selected):
            raw.append(samples[f"{prefix}_{select_idx}"]["y"][sample_idx])
        raw = np.array(raw)
        key = f"{prefix}_{select_idx}"
    else:
        key = f"{prefix}_0"
        raw = samples[key]["y"]

    return {"raw": raw, "time": samples[key]["x"], "source_ts": samples[key]["source_ts"]}


def extract_ground_truth(samples: dict, signal: dict, graph: GraphConfig) -> dict:
    """{"true_amplitude", "true_phase", "true_inst_freq"} of a simulated
    recording; all None for a real one.

    The signal is simulated either inside Falcon (SimulatedSource's meta
    channels) or by simulate_client.py, streaming through UDPSource and saving
    the truth to simulated_signal.npy. In the latter case, `signal` (from
    extract_selected_signal()) is trimmed in place to the file's length and its
    source_ts replaced by the simulator's.
    """
    if "UDPSource" in graph.processors and os.path.exists("simulated_signal.npy"):
        raw, time = signal["raw"], signal["time"]
        loaded = np.load("simulated_signal.npy")
        # UDPSource discards the first calib_packets for start-time calibration,
        # so drop the same rows here to line `loaded` back up with raw/time
        calib_packets = graph.options("UDPSource").get("calib_packets", 0)
        loaded = loaded[loaded["sample_counter"] >= calib_packets]
        assert raw[0, 0] == loaded["value"][0], \
            "Loaded simulated signal does not match UDPSource signal " \
            "(check calib_packets and that simulate_client.py was started after Falcon)"
        # simulated_signal.npy isn't covered by load_processor_signals' trim,
        # so ensure all arrays end up the same length here
        n = min(len(raw), len(time), len(loaded["phase"]), len(loaded["inst_freq"]),
                len(loaded["amplitude"]), len(loaded["source_ts"]))
        signal["raw"], signal["time"], signal["source_ts"] = raw[:n], time[:n], loaded["source_ts"][:n]
        return {
            "true_amplitude": loaded["amplitude"][:n],
            "true_phase":     np.angle(np.exp(1j * loaded["phase"][:n])),
            "true_inst_freq": loaded["inst_freq"][:n],
        }

    if "UDPSource" not in graph.processors and "SimulatedSource" in graph.processors:
        meta_phase = samples.get("SimulatedSource_meta_1", {}).get("y")
        return {
            "true_amplitude": samples.get("SimulatedSource_meta_0", {}).get("y"),
            "true_phase":     np.angle(np.exp(1j * meta_phase)) if meta_phase is not None else None,
            "true_inst_freq": samples.get("SimulatedSource_meta_2", {}).get("y"),
        }

    return {"true_amplitude": None, "true_phase": None, "true_inst_freq": None}
