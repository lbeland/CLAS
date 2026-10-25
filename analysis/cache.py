"""
Signal cache of a results folder. The first analysis parses the processors'
.bin files (clas_file_reader.py) and writes raw_signals.edf +
runtime_metadata.h5 (results_io.py); later runs load those two files instead,
which is much faster.

Either way, load_signals() returns the same Recording, so downstream code
doesn't need to care whether the data came from the .bin files or the cache.
"""

import os
from dataclasses import dataclass

import mne
import numpy as np

from .clas_file_reader import GraphConfig, load_processor_signals, extract_selected_signal, \
    extract_ground_truth, load_stim_annotations
from .results_io import write_raw_signals_edf, load_raw_signals_edf, \
    write_runtime_metadata, load_runtime_metadata

RAW_EDF_NAME = "raw_signals.edf"
META_H5_NAME = "runtime_metadata.h5"


@dataclass
class Recording:
    """The signals of one results folder."""
    samples:      dict                       # every processor output, keyed by label
    raw:          np.ndarray                 # selected channel (see extract_selected_signal())
    time:         np.ndarray                 # shared time axis [us]
    source_ts:    "np.ndarray | None"        # source timestamps of raw [us]
    ground_truth: dict                       # true_* fields; values None for real recordings
    annotations:  mne.Annotations | None     # stim protocol log events
    from_cache:   bool


def load_signals(results_dir: str, graph: GraphConfig, write_cache: bool = True) -> "Recording | None":
    """Load a results folder's signals from the cache if present, else from
    the .bin files (then writing the cache, unless write_cache is False).

    Returns:
        The Recording, or None if no source signal was found.
    """
    raw_edf_path = os.path.join(results_dir, RAW_EDF_NAME)
    meta_h5_path = os.path.join(results_dir, META_H5_NAME)

    if os.path.exists(raw_edf_path) and os.path.exists(meta_h5_path):
        return load_cache(raw_edf_path, meta_h5_path, graph)

    samples = load_processor_signals(graph.fs, results_dir, list(graph.processors))
    signal  = extract_selected_signal(samples, graph)
    if signal is None:
        print("Error: no source signal found (UDPSource or SimulatedSource).")
        return None
    ground_truth = extract_ground_truth(samples, signal, graph)

    annotations = load_stim_annotations(results_dir, signal["time"][0])
    if write_cache:
        write_raw_signals_edf(raw_edf_path, graph.fs, samples, signal["time"], annotations=annotations)
        write_runtime_metadata(meta_h5_path, graph.fs, samples, signal["time"], ground_truth)
    return Recording(samples=samples, **signal, ground_truth=ground_truth,
                     annotations=annotations, from_cache=False)


def load_cache(raw_edf_path: str, meta_h5_path: str, graph: GraphConfig) -> Recording:
    """Combine raw_signals.edf + runtime_metadata.h5 back into a Recording,
    with the same samples keys as load_processor_signals()."""
    meta = load_runtime_metadata(meta_h5_path)
    edf  = load_raw_signals_edf(raw_edf_path)

    fs   = meta["fs"]
    # Authoritative over edf["n"]: EDF pads the final record to a whole second
    n    = meta["n_samples"]
    time = meta["start_ts"] + np.round(np.arange(n) / fs * 1e6).astype(np.int64)

    def source_ts(key):
        return meta["source_ts"].get(key)

    # Only the lowest-indexed recorded channel carries hardware_ts (see write_runtime_metadata)
    hw_ts = meta.get("hardware_ts")
    hw_ts = hw_ts[:n] if hw_ts is not None else None
    first_idx = min(edf["eeg"], default=None)

    raw_prefix = meta["raw_source_class"]
    samples = {}
    for idx, y in edf["eeg"].items():
        samples[f"{raw_prefix}_{idx}"] = {"x": time, "y": y[:n], "source_ts": source_ts(f"{raw_prefix}_{idx}")}
        if idx == first_idx and hw_ts is not None:
            samples[f"{raw_prefix}_{idx}"]["hw_ts"] = hw_ts
    if edf["trigger"] is not None:
        samples["UDPSource_TRIGGER"] = {"x": time, "y": edf["trigger"][:n], "source_ts": source_ts("UDPSource_TRIGGER")}

    online = meta["online"]
    if "stimulus" in online:
        samples["StimControl"] = {"x": time, "y": online["stimulus"], "source_ts": source_ts("StimControl")}
    if "f0" in online:
        samples["FrequencyEstimation"] = {"x": time, "y": online["f0"], "source_ts": source_ts("FrequencyEstimation")}
    if "f0_raw" in online:
        samples["FrequencyEstimation_raw"] = {"x": time, "y": online["f0_raw"], "source_ts": source_ts("FrequencyEstimation_raw")}
    if "phase" in online:
        samples["PhaseEstimation_phase"] = {"x": time, "y": online["phase"], "source_ts": source_ts("PhaseEstimation_phase")}
    if "filt" in online:
        samples["ecHTFilter"] = {"x": time, "y": online["filt"], "source_ts": source_ts("ecHTFilter")}
    if "channel_idx" in online:
        samples["ChannelSelection"] = {"x": time, "y": online["channel_idx"], "source_ts": source_ts("ChannelSelection")}

    signal = extract_selected_signal(samples, graph)
    if signal is None:
        raise ValueError(f"{raw_edf_path}: could not reconstruct the selected signal (no source in the graph).")
    ground_truth = extract_ground_truth(samples, signal, graph)

    sim = meta["simulation"]
    for field in ("true_amplitude", "true_phase", "true_inst_freq"):
        if field in sim:
            ground_truth[field] = sim[field]

    return Recording(samples=samples, **signal, ground_truth=ground_truth,
                     annotations=edf["annotations"], from_cache=True)
