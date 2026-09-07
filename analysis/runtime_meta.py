"""
runtime_metadata.h5: pipeline-internal data that has no business in a
generic, tool-agnostic EDF (raw hardware timestamps, online f0/phase/
filter estimates, the selected-channel index, simulation-only ground
truth). Written once alongside raw_signals.edf (see edf_io.py) and read
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

ONLINE_FIELDS = {
    "StimControl":  "stimulus",
    "FrequencyEstimation":        "f0",
    "PhaseEstimation_phase": "phase",
    "ecHTFilter":           "filt",
    "ChannelSelection":      "channel_idx",
}

SIMULATION_FIELDS = ("true_amplitude", "true_phase", "true_inst_freq")


def _normalize_raw_channel_key(key: str) -> str:
    """Raw per-channel keys come from either UDPSource_{idx} (real
    hardware) or SimulatedSource_{idx} (simulated runs); edf_io.write_raw_signals_edf
    writes both the same way and edf_io.load_runtime() always reconstructs
    them as UDPSource_{idx} on reload, so source_ts must be stored under
    that same normalized name or the reload lookup silently misses it."""
    if key.startswith("SimulatedSource_") and key[len("SimulatedSource_"):].isdigit():
        return f"UDPSource_{key[len('SimulatedSource_'):]}"
    return key


def write_runtime_metadata(filepath: str, fs: float, samples: dict, ground_truth: dict) -> None:
    with h5py.File(filepath, "w") as f:
        f.attrs["fs"] = fs
        f.attrs["start_ts"] = int(ground_truth["time"][0])
        # mne's EDF writer pads the final data record up to a whole number of
        # seconds, so raw_signals.edf's channels usually come back a few
        # samples longer than they went in. Keep the true count here so
        # load_runtime() can truncate back to it -- otherwise the (padded)
        # EEG/AUX/Trigger arrays and the (exact) online-estimate arrays below
        # end up different lengths after a cached reload.
        f.attrs["n_samples"] = len(ground_truth["time"])

        src_grp = f.create_group("source_ts")
        for key, entry in samples.items():
            ts = entry.get("source_ts")
            if ts is not None:
                src_grp.create_dataset(_normalize_raw_channel_key(key), data=np.asarray(ts))

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
    """Returns {"fs", "start_ts", "source_ts": {key: array}, "online": {field: array},
    "simulation": {field: array}} -- the latter three dicts only contain keys that
    were actually present at write time."""
    meta = {"source_ts": {}, "online": {}, "simulation": {}}
    with h5py.File(filepath, "r") as f:
        meta["fs"] = float(f.attrs["fs"])
        meta["start_ts"] = int(f.attrs["start_ts"])
        meta["n_samples"] = int(f.attrs["n_samples"])
        for key in f.get("source_ts", {}):
            meta["source_ts"][key] = f["source_ts"][key][()]
        for key in f.get("online", {}):
            meta["online"][key] = f["online"][key][()]
        for key in f.get("simulation", {}):
            meta["simulation"][key] = f["simulation"][key][()]
    return meta
