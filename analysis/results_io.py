"""
Results-folder files: two EDFs for viewing and one HDF5 file for pipeline data.

EDF files
---------
- raw_signals.edf: a generic, tool-agnostic recording (EEG channels, AUX,
  Trigger, stim-log annotations) that any EDF reader can open -- no
  CLAS-pipeline-specific data. Written once from the processors' .bin
  output, together with runtime_metadata.h5 (see cache.py).
- analysis.edf: the offline-analysis outputs (selected-channel raw,
  offline+online filter/phase/f0, target+trigger stimulus), for visually
  comparing online vs. offline estimates in an EDF viewer. Cheap to
  rebuild every run from the cached raw_signals.edf + runtime_metadata.h5.

HDF5 metadata
-------------
- runtime_metadata.h5: pipeline-internal data (raw hardware timestamps,
  the selected-channel index, simulation-only ground truth).
  Written once alongside raw_signals.edf (see cache.py) and
  read back on every later run.

  Datasets are gzip-compressed and identical arrays are stored once (hard
  links).
"""


import datetime
import warnings

import mne
import numpy as np
import pytz
import hashlib
import h5py

from .clas_file_reader import CHANNEL_NAMES, first_udp_channel_key

_INDEX_TO_NAME = CHANNEL_NAMES  # {1-based channel index: electrode name, e.g. 3: "Fz"}
_NAME_TO_INDEX = {name: idx for idx, name in CHANNEL_NAMES.items()}

ONLINE_FIELDS = {
    "StimControl":  "stimulus",
    "FrequencyEstimation":        "f0",
    "FrequencyEstimation_raw":    "f0_raw",  # optional: raw pre-Kalman peak f0 (debug_output=true)
    "PhaseEstimation_phase": "phase",
    "ecHTFilter":           "filt",
    "ChannelSelection":      "channel_idx",
}

SIMULATION_FIELDS = ("true_amplitude", "true_phase", "true_inst_freq")

_COMPRESSION = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def _write_edf(filepath: str, fs: float, start_time_us: float, channels: list[tuple],
                annotations: "mne.Annotations | None" = None) -> None:
    """Write channels to an EDF file.

    Args:
        channels: List of (name, mne_ch_type, values). Non-EEG numeric
            channels (phase, target stimulus, ...) must use ch_type "stim",
            not "misc": mne's EDF writer always labels non-stim/non-eeg
            channels' physical_dimension as "uV", and mne.io.read_raw_edf()
            then divides "misc" channels by 1e6 on read; "stim" channels
            round-trip exactly instead.

    add_ch_type=False keeps channel labels as given, within EDF's
    16-character label limit.
    """
    min_len = min(len(ch[2]) for ch in channels)
    info    = mne.create_info([ch[0] for ch in channels], sfreq=fs,
                              ch_types=[ch[1] for ch in channels], verbose=False)
    raw_mne = mne.io.RawArray([ch[2][:min_len] for ch in channels], info, verbose=False)
    raw_mne.apply_function(lambda x: x * 1e-6, picks="eeg")
    start_dt = datetime.datetime.fromtimestamp(
        start_time_us / 1e6, tz=pytz.timezone("Europe/Berlin")
    ).replace(tzinfo=datetime.timezone.utc)
    raw_mne.set_meas_date(start_dt)
    if annotations is not None:
        # Drop orig_time so onsets stay relative to the first sample, not shifted by start_dt's sub-second part.
        # Drop BAD_ACQ_SKIP (read back from a previous export's padding); export re-adds it.
        keep = annotations.description != "BAD_ACQ_SKIP"
        raw_mne.set_annotations(mne.Annotations(annotations.onset[keep], annotations.duration[keep],
                                                annotations.description[keep]))
    # EDF needs equal-length data records, so mne pads the last one and warns
    # about it; that is expected, so silence just that warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="EDF format requires equal-length data blocks")
        raw_mne.export(filepath, fmt="edf", add_ch_type=False,
                       physical_range="channelwise", overwrite=True, verbose=False)


def _resolve_channel(raw: "mne.io.BaseRaw", name: str) -> str | None:
    """Match a channel by name. Channel labels have no type prefix (see
    _write_edf's add_ch_type=False), so this is just a presence check."""
    return name if name in raw.ch_names else None


# ---------------------------------------------------------------------------
# raw_signals.edf
# ---------------------------------------------------------------------------

def write_raw_signals_edf(
    filepath: str,
    fs: float,
    samples: dict,
    time: np.ndarray,
    annotations: "mne.Annotations | None" = None,
) -> None:
    """Write the generic, tool-agnostic recording (EEG channels, AUX, Trigger)
    to raw_signals.edf.

    `time` is the recording's time axis [us]; channels are trimmed to its length.

    Channels with a known 10-20 electrode (clas_file_reader.CHANNEL_NAMES, e.g.
    index 3 -> "Fz") are labelled with that name; others get "EEG_{index}".
    Only real "UDPSource_*" channels get an electrode name -- synthetic
    "SimulatedSource_*" channels always get "EEG_{idx+1}", since load_cache()
    reconstructs either as "UDPSource_{idx}" on reload regardless.
    """
    n = len(time)

    channels = []
    eeg_keys = sorted(
        k for k in samples
        if (k.startswith("UDPSource_") or k.startswith("SimulatedSource_")) and k.split("_")[1].isdigit()
    )
    for key in eeg_keys:
        ch_idx = int(key.split("_")[1])
        ch_num = ch_idx + 1  # 1-based, matches CHANNEL_NAMES
        if key.startswith("SimulatedSource_"):
            label = f"EEG_{ch_num}"
        else:
            label = _INDEX_TO_NAME.get(ch_num, f"EEG_{ch_num}")
        channels.append((label, "eeg", samples[key]["y"][:n]))

    # AUX_0..AUX_7 are recorded analog channels from the same ADC as the EEG
    # channels, so they get the same "eeg" type/uV scaling
    for idx in range(8):
        aux = samples.get(f"UDPSource_AUX_{idx}")
        if aux is not None:
            channels.append((f"AUX_{idx}", "eeg", aux["y"][:n]))
    if samples.get("UDPSource_TRIGGER") is not None:
        channels.append(("Trigger", "stim", samples["UDPSource_TRIGGER"]["y"][:n]))

    _write_edf(filepath, fs, time[0], channels, annotations)


def load_raw_signals_edf(filepath: str) -> dict:
    """Read raw_signals.edf back into {"eeg": {idx: y}, "trigger": y|None,
    "n": n, "annotations": Annotations}.

    No time axis is reconstructed here: EDF's header only stores the
    recording start time to 1-second resolution, so runtime_metadata.h5's
    start_ts attribute (exact, see cache.load_cache()) is the authoritative one."""
    raw_mne = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    n = len(raw_mne.times)

    eeg = {}
    for ch in raw_mne.ch_names:
        if ch in _NAME_TO_INDEX:
            ch_num = _NAME_TO_INDEX[ch]
        elif ch.startswith("EEG_"):
            ch_num = int(ch.split("_")[1])
        else:
            continue
        eeg[ch_num - 1] = raw_mne.get_data(picks=ch)[0] * 1e6

    trig_ch = _resolve_channel(raw_mne, "Trigger")
    trigger = raw_mne.get_data(picks=trig_ch)[0] if trig_ch is not None else None

    return {"eeg": eeg, "trigger": trigger, "n": n, "annotations": raw_mne.annotations}


# ---------------------------------------------------------------------------
# analysis.edf
# ---------------------------------------------------------------------------

def write_analysis_edf(
    filepath: str,
    fs: float,
    raw: np.ndarray,
    time: np.ndarray,
    samples: dict,
    filtered: np.ndarray = None,
    hilbert_phase: np.ndarray = None,
    f0_continuous: np.ndarray = None,
    stim_ref: np.ndarray = None,
    annotations: "mne.Annotations | None" = None,
    whitened: bool = False,
) -> None:
    """Write the offline-analysis outputs (selected-channel raw, offline/online
    filter+phase+f0, target+trigger stimulus) to analysis.edf, for visually
    comparing online vs. offline estimates in an EDF viewer.

    Args:
        raw: Selected-channel signal; time: its time axis [us].
        whitened: Whether `filtered` was spectrally whitened (see
            compute_hilbert_reference); if so, it's on a different scale than
            Raw's uV and is stored as "stim" instead of "eeg".
    """
    n = len(raw)

    channels = [("Raw", "eeg", raw)]

    if filtered is not None:
        channels.append(("Filt_off", "stim" if whitened else "eeg", filtered[:n]))
    if samples.get("ecHTFilter") is not None:
        # Plain online bandpass of raw uV EEG, no whitening: "eeg" like Raw so both stay comparable
        channels.append(("Filt_on", "eeg", samples["ecHTFilter"]["y"][:n]))
    if hilbert_phase is not None:
        channels.append(("Phase_off", "stim", np.nan_to_num(hilbert_phase, nan=-2 * np.pi)[:n]))
    if samples.get("PhaseEstimation_phase") is not None:
        phase_est = np.nan_to_num(samples["PhaseEstimation_phase"]["y"], nan=-2 * np.pi)
        channels.append(("Phase_on", "stim", phase_est[:n]))
    if f0_continuous is not None:
        channels.append(("F0_off_Hz", "stim", np.nan_to_num(f0_continuous)[:n]))
    if samples.get("FrequencyEstimation") is not None:
        channels.append(("F0_on_Hz", "stim", np.nan_to_num(samples["FrequencyEstimation"]["y"])[:n]))
    if stim_ref is not None:
        channels.append(("Target_Stim", "stim", stim_ref[:n]))

    trigger = samples.get("UDPSource_TRIGGER")
    if trigger is not None:
        channels.append(("Trigger_hw", "stim", trigger["y"][:n]))
    trigger_sw = samples.get("StimControl")
    if trigger_sw is not None:
        channels.append(("Trigger_sw", "stim", trigger_sw["y"][:n]))

    _write_edf(filepath, fs, time[0], channels, annotations)


def _dataset_writer(f: h5py.File):
    """Returns put(key, data): writes a compressed dataset, or hard-links an identical one."""
    stored = {}

    def put(key, data):
        data   = np.asarray(data)
        digest = (data.dtype.str, data.shape, hashlib.sha1(data.tobytes()).digest())
        if digest in stored:
            f[key] = f[stored[digest]]
        else:
            f.create_dataset(key, data=data, **_COMPRESSION)
            stored[digest] = key

    return put


def write_runtime_metadata(filepath: str, fs: float, samples: dict, time: np.ndarray,
                           ground_truth: dict) -> None:
    """`time` is the recording's time axis [us], ground_truth its true_* fields."""
    with h5py.File(filepath, "w") as f:
        put = _dataset_writer(f)
        f.attrs["fs"] = fs
        f.attrs["start_ts"] = int(time[0])
        # True count, so load_cache() can truncate raw_signals.edf's
        # record-padded channels back to match the exact arrays below
        f.attrs["n_samples"] = len(time)
        # Raw per-channel key prefix actually used ("UDPSource" or
        # "SimulatedSource"), so load_cache() can reconstruct the same keys
        f.attrs["raw_source_class"] = (
            "SimulatedSource" if any(k.startswith("SimulatedSource_") for k in samples)
            else "UDPSource"
        )
        # UDPSource's reconstructed ADC sampling times, needed only for its
        # hardware->source latency calc in latency.compute_latencies()
        udp = samples.get(first_udp_channel_key(samples), {})
        if udp.get("hw_ts") is not None:
            put("hardware_ts", udp["hw_ts"])

        f.create_group("source_ts")
        for key, entry in samples.items():
            ts = entry.get("source_ts")
            if ts is not None:
                put(f"source_ts/{key}", ts)

        f.create_group("online")
        for samples_key, field in ONLINE_FIELDS.items():
            if samples.get(samples_key) is not None:
                put(f"online/{field}", samples[samples_key]["y"])

        if any(ground_truth.get(k) is not None for k in SIMULATION_FIELDS):
            f.create_group("simulation")
            for field in SIMULATION_FIELDS:
                if ground_truth.get(field) is not None:
                    put(f"simulation/{field}", ground_truth[field])


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
