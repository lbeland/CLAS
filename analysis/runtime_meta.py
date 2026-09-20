"""
runtime_metadata.h5: pipeline-internal data that has no business in a
generic, tool-agnostic EDF (raw hardware timestamps, the selected-channel index,
simulation-only ground truth). Written once alongside raw_signals.edf (see edf_io.py) and read
back on every later run.

HDF5's native typing sidesteps three real EDF limitations discovered while
building the single-EDF version of this cache: the header only stores the
recording start time to 1-second resolution (any sub-second offset was
silently truncated), a channel typed "misc" gets a "uV" physical-dimension
label from mne's writer regardless of its actual unit and is then silently
divided by 1e6 by mne's reader, and 16-bit digital quantization perturbs
exact integers (e.g. a constant selected-channel index) enough to break
dict-key lookups downstream.
"""

import h5py
import numpy as np

from .loader import first_udp_channel_key

ONLINE_FIELDS = {
    "StimControl":  "stimulus",
    "FrequencyEstimation":        "f0",
    "FrequencyEstimation_raw":    "f0_raw",  # optional: raw pre-Kalman peak f0 (debug_output=true)
    "PhaseEstimation_phase": "phase",
    "ecHTFilter":           "filt",
    "ChannelSelection":      "channel_idx",
}

SIMULATION_FIELDS = ("true_amplitude", "true_phase", "true_inst_freq")


def write_runtime_metadata(filepath: str, fs: float, samples: dict, ground_truth: dict) -> None:
    with h5py.File(filepath, "w") as f:
        f.attrs["fs"] = fs
        f.attrs["start_ts"] = int(ground_truth["time"][0])
        # True count, so load_runtime() can truncate raw_signals.edf's
        # record-padded channels back to match the exact arrays below
        f.attrs["n_samples"] = len(ground_truth["time"])
        # Raw per-channel key prefix actually used ("UDPSource" or
        # "SimulatedSource"), so load_runtime() can reconstruct the same keys
        f.attrs["raw_source_class"] = (
            "SimulatedSource" if any(k.startswith("SimulatedSource_") for k in samples)
            else "UDPSource"
        )
        # UDPSource's reconstructed ADC sampling times, needed only for its
        # hardware->source latency calc in analyse_latencies()
        udp = samples.get(first_udp_channel_key(samples), {})
        if udp.get("hw_ts") is not None:
            f.create_dataset("hardware_ts", data=np.asarray(udp["hw_ts"]))

        src_grp = f.create_group("source_ts")
        for key, entry in samples.items():
            ts = entry.get("source_ts")
            if ts is not None:
                src_grp.create_dataset(key, data=np.asarray(ts))

        online = f.create_group("online")
        for samples_key, field in ONLINE_FIELDS.items():
            if samples.get(samples_key) is not None:
                online.create_dataset(field, data=samples[samples_key]["y"])

        if any(ground_truth.get(k) is not None for k in SIMULATION_FIELDS):
            sim = f.create_group("simulation")
            for field in SIMULATION_FIELDS:
                if ground_truth.get(field) is not None:
                    sim.create_dataset(field, data=np.asarray(ground_truth[field]))

    print(f"HDF5 written: {filepath}")


def load_runtime_metadata(filepath: str) -> dict:
    """Returns {"fs", "start_ts", "n_samples", "raw_source_class",
    "hardware_ts": array | absent, "source_ts": {key: array},
    "online": {field: array}, "simulation": {field: array}}.

    "hardware_ts" is absent for caches written before it started being saved;
    "raw_source_class" defaults to "UDPSource" for caches written before it
    was added. The last three dicts only contain keys actually present at
    write time.
    """
    meta = {"source_ts": {}, "online": {}, "simulation": {}}
    with h5py.File(filepath, "r") as f:
        meta["fs"] = float(f.attrs["fs"])
        meta["start_ts"] = int(f.attrs["start_ts"])
        meta["n_samples"] = int(f.attrs["n_samples"])
        meta["raw_source_class"] = f.attrs.get("raw_source_class", "UDPSource")
        if "hardware_ts" in f:
            meta["hardware_ts"] = f["hardware_ts"][()]
        for key in f.get("source_ts", {}):
            meta["source_ts"][key] = f["source_ts"][key][()]
        for key in f.get("online", {}):
            meta["online"][key] = f["online"][key][()]
        for key in f.get("simulation", {}):
            meta["simulation"][key] = f["simulation"][key][()]
    return meta
