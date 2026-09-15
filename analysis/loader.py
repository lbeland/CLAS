"""
Data loading: file discovery, processor signal loading, ground-truth extraction,
graph structure parsing, and latency reporting.
"""

import glob
import os
import numpy as np
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt

from .read_output import get_signal_data
from .stimulus import get_edges
from .plot import FIGSIZE, save_pgf, save_pdf, PLOTS_DIR

RESULTS_DIR = "_last_run"


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
    for key in samples:
        samples[key]["x"] = samples[key]["x"][:min_lengths]
        samples[key]["y"] = samples[key]["y"][:min_lengths]
        if samples[key].get("source_ts") is not None:
            samples[key]["source_ts"] = samples[key]["source_ts"][:min_lengths]

    return samples


def extract_ground_truth(samples: dict, graph_config: dict | None = None) -> dict | None:
    """Extract ground-truth signals from UDPSource or SimulatedSource output."""
    if "UDPSource_0" in samples:
        if "ChannelSelection" in samples:
            channel_selected = samples["ChannelSelection"]["y"] - 1  # convert to zero-based
            raw = []
            for sample_idx, select_idx in enumerate(channel_selected):
                raw.append(samples[f"UDPSource_{select_idx}"]["y"][sample_idx])
            raw = np.array(raw)
            time        = samples[f"UDPSource_{select_idx}"]["x"]
            source_time = samples[f"UDPSource_{select_idx}"]["source_ts"]
        else:
            raw         = samples["UDPSource_0"]["y"]
            time        = samples["UDPSource_0"]["x"]
            source_time = samples["UDPSource_0"]["source_ts"]

        if os.path.exists("simulated_signal.npy"):
            loaded = np.load("simulated_signal.npy")
            # UDPSource discards the first `calib_packets` packets for start-time
            # calibration before it publishes anything (see UDPSource::Process's
            # calibration phase), so Falcon's first recorded sample is
            # simulate_client.py's sample `calib_packets`, not sample 0. Drop
            # the same number of rows here so `loaded` lines back up with
            # `raw`/`time`. This assumes simulate_client.py was only started
            # once UDPSource was already bound and listening (see README) --
            # otherwise packets sent before that are lost on the wire and this
            # count would be off by however many were dropped.
            calib_packets = (
                (graph_config or {})
                .get("graph", {}).get("processors", {})
                .get("UDPSource", {}).get("options", {})
                .get("calib_packets", 0)
            )
            loaded = loaded[loaded["sample_counter"] >= calib_packets]
            assert raw[0, 0] == loaded["value"][0], \
                "Loaded simulated signal does not match UDPSource signal " \
                "(check calib_packets and that simulate_client.py was started after Falcon)"
            # simulated_signal.npy is generated independently of the recorded
            # run and isn't covered by load_processor_signals' min_lengths
            # trim, so it can be longer/shorter than raw/time -- trim
            # everything here to a common length so downstream code can rely
            # on ground_truth's arrays always matching in length.
            n = min(len(raw), len(time), len(loaded["phase"]), len(loaded["inst_freq"]),
                    len(loaded["amplitude"]), len(loaded["source_ts"]))
            return {
                "raw":            raw[:n],
                "time":           time[:n],
                "true_amplitude": loaded["amplitude"][:n],
                "true_phase":     np.angle(np.exp(1j * loaded["phase"][:n])),
                "true_inst_freq": loaded["inst_freq"][:n],
                "source_ts":      loaded["source_ts"][:n],
            }
        return {
            "raw": raw, "time": time, "source_ts": source_time,
            "true_amplitude": None, "true_phase": None, "true_inst_freq": None,
        }

    elif "SimulatedSource_0" in samples:
        if "ChannelSelection" in samples:
            channel_selected = samples["ChannelSelection"]["y"]
            raw = []
            for sample_idx, select_idx in enumerate(channel_selected):
                raw.append(samples[f"SimulatedSource_{select_idx-1}"]["y"][sample_idx])
            raw         = np.array(raw)
            time        = samples[f"SimulatedSource_{select_idx-1}"]["x"]
            source_time = samples[f"SimulatedSource_{select_idx-1}"]["source_ts"]
        else:
            raw         = samples["SimulatedSource_0"]["y"]
            time        = samples["SimulatedSource_0"]["x"]
            source_time = samples["SimulatedSource_0"]["source_ts"]

        return {
            "raw":            raw,
            "time":           time,
            "source_ts":      source_time,
            "true_amplitude": samples.get("SimulatedSource_meta_0", {}).get("y"),
            "true_phase":     np.angle(np.exp(1j * samples["SimulatedSource_meta_1"]["y"])),
            "true_inst_freq": samples.get("SimulatedSource_meta_2", {}).get("y"),
        }

    return None


# ---------------------------------------------------------------------------
# Graph structure + latency
# ---------------------------------------------------------------------------

def _parse_graph_structure(graph_config: dict) -> tuple[dict, list]:
    """Parse graph connections → (predecessors dict, topological order).
    Serializers and BenchSinks are excluded."""
    processors  = graph_config.get("graph", {}).get("processors", {})
    connections = graph_config.get("graph", {}).get("connections", [])

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


def _get_source_ts(proc_name: str, samples: dict, ground_truth: dict):
    """Resolve source_ts array for a processor by its yaml name."""
    if proc_name in ("UDPSource", "SimulatedSource"):
        return ground_truth.get("source_ts")
    key  = f"{proc_name}_phase" if proc_name == "PhaseEstimation" else proc_name
    data = samples.get(key)
    return data.get("source_ts") if data else None


def _write_latency_table(pipeline_rows: "list[dict]", audio_rows: "list[dict]",
                         name: str = "latency_table",
                         caption: str = "Pipeline-stage latency [$\\mu$s] and audio onset "
                                        "latency [ms], mean/median/std across samples plus "
                                        "the single worst case (its index in brackets).",
                         label: str = "tab:latency_table") -> None:
    """Write pipeline-stage + audio-onset latency to one combined LaTeX table
    at PLOTS_DIR/<name>.tex, matching the plot.py/sweep.py table convention.

    Each row dict is {"from", "to", "mean", "median", "std", "max", "idx"};
    "from"/"to" are merged into one "From $\\to$ To" cell. A row with
    "total"=True (the pipeline's source->last_proc row) is set off with a
    \\midrule and that whole merged cell bolded. The two row groups are
    separated by one empty row (mixed units -- see caption -- so this is a
    visual gap, not a \\midrule)."""
    def _esc(text):
        return str(text).replace("_", r"\_")

    def _cell(row):
        max_cell = f"{row['max']:.2f} (idx={row['idx']})" if np.isfinite(row['max']) else "--"
        stage_cell = rf"{_esc(row['from'])} $\to$ {_esc(row['to'])}"
        if row.get("total"):
            stage_cell = rf"\textbf{{Total: {stage_cell}}}"
        return [stage_cell,
                f"{row['mean']:.2f}", f"{row['median']:.2f}", f"{row['std']:.2f}", max_cell]

    body = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"From $\to$ To & Mean & Median & Std & Max (idx) \\", r"\midrule"]
    for row in pipeline_rows:
        body.append(" & ".join(_cell(row)) + r" \\")
        if row.get("total"):
            body.append(r"\midrule")
    if pipeline_rows and audio_rows:
        body.append(r" & & & & \\")  # empty row: visual gap between the two (differently-unitted) groups
    for row in audio_rows:
        body.append(" & ".join(_cell(row)) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]

    tex = [r"\begin{table}[htbp]", r"  \centering", rf"  \caption{{{caption}}}",
           rf"  \label{{{label}}}", *("  " + ln for ln in body), r"\end{table}"]
    tex_path = os.path.join(PLOTS_DIR, f"{name}.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"wrote {tex_path}")


def compute_total_latency(samples: dict, ground_truth: dict, graph_config: dict) -> "np.ndarray | None":
    """End-to-end pipeline latency [us]: per-sample source_ts of the last
    processor in topological order minus source_ts of the source processor
    (UDPSource/SimulatedSource), matched by sample index.

    Shared by analyse_latencies() (which additionally breaks this down
    stage-by-stage, prints/plots it, and writes a LaTeX table) and pooled
    analyses that only want the end-to-end number, e.g.
    analyse_pooled_errors() in main.py. Returns None if the graph doesn't
    have at least two processors with source_ts data.
    """
    _, topo_order = _parse_graph_structure(graph_config)
    proc_ts = {p: _get_source_ts(p, samples, ground_truth) for p in topo_order}
    proc_ts = {p: ts for p, ts in proc_ts.items() if ts is not None}
    if len(proc_ts) < 2:
        return None

    source_proc = next((p for p in topo_order if p in proc_ts), None)
    last_proc   = next((p for p in reversed(topo_order) if p in proc_ts), None)
    if not source_proc or not last_proc or last_proc == source_proc:
        return None

    ts_start = proc_ts[source_proc]
    ts_end   = proc_ts[last_proc]
    n = min(len(ts_start), len(ts_end))
    return ts_end[:n] - ts_start[:n]


def analyse_latencies(samples: dict, ground_truth: dict, graph_config: dict) -> None:
    """Print a per-stage latency table derived from source_timestamps, and
    write it (pipeline stages + audio onset, combined) to a LaTeX .tex file
    via _write_latency_table()."""
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
    
    plt.figure(figsize=(15, 4))

    pipeline_rows: list[dict] = []
    audio_rows: list[dict] = []

    col = 22
    W   = col * 2 + 52
    print(f"\n{'─' * W}")
    print(f"  Pipeline latency analysis (us)")
    print(f"{'─' * W}")
    print(f"  {'From':<{col}} {'To':<{col}} {'Mean':>8} {'Median':>8} {'Std':>8} {'Max':>8} {'Idx':>8}")
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
        lat = (ts_proc[:n] - ts_ancestor[:n])
        plt.plot(lat, alpha=0.5, linewidth=0.5, label=f"{ancestor} → {proc}")
        print(f"  {ancestor:<{col}} {proc:<{col}} "
              f"{np.mean(lat):>8.2f} {np.median(lat):>8.2f} "
              f"{np.std(lat):>8.2f} {np.max(lat):>8.2f} {np.argmax(lat):>8}")
        pipeline_rows.append({"from": ancestor, "to": proc,
                              "mean": np.mean(lat), "median": np.median(lat),
                              "std": np.std(lat), "max": np.max(lat), "idx": np.argmax(lat)})

    last_proc = next((p for p in reversed(topo_order) if p in proc_ts), None)
    total = compute_total_latency(samples, ground_truth, graph_config)
    if source_proc and last_proc and last_proc != source_proc and total is not None:
        plt.plot(total, alpha=0.5, linewidth=0.5, label=f"{source_proc} → {last_proc}", color="black")
        print(f"{'─' * W}")
        print(f"  {'TOTAL  ' + source_proc:<{col}} {last_proc:<{col}} "
              f"{np.mean(total):>8.2f} {np.median(total):>8.2f} "
              f"{np.std(total):>8.2f} {np.max(total):>8.2f} {np.argmax(total):>8}")
        pipeline_rows.append({"from": source_proc, "to": last_proc, "total": True,
                              "mean": np.mean(total), "median": np.median(total),
                              "std": np.std(total), "max": np.max(total), "idx": np.argmax(total)})
    print(f"{'─' * W}\n")

    plt.legend(loc="upper left", fontsize=9)
    plt.xlabel("Sample Index")
    plt.ylabel(r"Latency [$\mu$s]")
    plt.ylim(0, 1000)
    # plt.savefig("latency_analysis.png", dpi=300, bbox_inches="tight")

    # Latency between computed stimulus onset (StimControl) and the
    # recorded trigger onset (UDPSource_TRIGGER), matched edge-by-edge.
    if "StimControl" in samples and "UDPSource_TRIGGER" in samples:
        fig = plt.figure(figsize=FIGSIZE)
        stim = samples["StimControl"]
        trig = samples["UDPSource_TRIGGER"]

        stim_edges = get_edges((stim["y"] > 0.5).astype(float), "rising")
        trig_edges = get_edges((trig["y"] > 0.5).astype(float), "rising")

        if len(stim_edges) == 0 or len(trig_edges) == 0:
            print("Not enough stimulus/trigger edges for onset latency analysis.")
        elif stim.get("source_ts") is None or trig.get("source_ts") is None:
            print("Missing source_ts for stimulus/trigger onset latency analysis.")
        else:
            if len(stim_edges) != len(trig_edges):
                print(f"Warning: stimulus/trigger edge count mismatch — "
                      f"stim={len(stim_edges)}, trigger={len(trig_edges)}. Matching by nearest preceding index.")

            # For each trigger edge, match the nearest stim edge at or before it
            # (searchsorted assumes stim_edges is sorted ascending, as returned
            # by get_edges).
            match_pos = np.searchsorted(stim_edges, trig_edges, side="right") - 1
            unmatched = match_pos < 0
            if np.any(unmatched):
                print(f"Warning: {np.sum(unmatched)} trigger edge(s) precede the first "
                      f"stimulus edge and have no valid match — dropping them.")

            stim_source_ts = stim["source_ts"]
            trig_source_ts = trig["source_ts"]
            trig_edges_matched = trig_edges[~unmatched]
            matched_stim_edges = stim_edges[match_pos[~unmatched]]
            latency_ms = (trig_source_ts[trig_edges_matched] - stim_source_ts[matched_stim_edges]) / 1e3

            plt.plot(latency_ms)
            plt.axhline(np.median(latency_ms), color="red", linestyle="--", label=f"Median: {np.median(latency_ms):.2f} ms")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            plt.xlabel("Stimulus index")
            plt.ylabel("Latency [ms]")
            plt.legend(loc="upper right")

            save_pdf(fig, "latency_analysis")
            save_pgf(fig, "latency_analysis")

            print(f" Audio onset latency (ms)")
            print(f"{'─' * W}")
            print(f"  {'StimControl':<{col}} {'UDPSource_TRIGGER':<{col}} "
                  f"{np.mean(latency_ms):>8.2f} {np.median(latency_ms):>8.2f} "
                  f"{np.std(latency_ms):>8.2f} {np.max(latency_ms):>8.2f} {np.argmax(latency_ms):>8}")
            print(f"{'─' * W}\n")
            audio_rows.append({"from": "StimControl", "to": "UDPSource_TRIGGER",
                               "mean": np.mean(latency_ms), "median": np.median(latency_ms),
                               "std": np.std(latency_ms), "max": np.max(latency_ms),
                               "idx": np.argmax(latency_ms)})

    if pipeline_rows or audio_rows:
        _write_latency_table(pipeline_rows, audio_rows)



