"""
Raw/metadata/analysis EDF+HDF5 split.

Each results folder gets three files:

- raw_signals.edf: a generic, tool-agnostic recording (EEG channels, AUX,
  Trigger, stim-log annotations) that any EDF reader can open -- no
  CLAS-pipeline-specific data. Written once from the processors' .bin
  output.
- runtime_metadata.h5: everything pipeline-internal that doesn't belong in
  a generic EDF (raw hardware timestamps, online f0/phase/filter,
  selected-channel index, simulation-only ground truth). See
  analysis/runtime_meta.py. Written alongside raw_signals.edf.
- analysis.edf: the offline-analysis outputs (selected-channel raw,
  offline+online filter/phase/f0, target+trigger stimulus), for visually
  comparing online vs. offline estimates in an EDF viewer. Cheap to
  rebuild every run from the cached raw_signals.edf + runtime_metadata.h5.

load_runtime() recombines raw_signals.edf + runtime_metadata.h5 into the
same `samples`/`ground_truth` shapes produced by
analysis.loader.load_processor_signals()/extract_ground_truth(), so
downstream code doesn't need to care whether the data came from the raw
.bin files or the cached files.
"""

import datetime

import mne
import numpy as np
import pytz

from .loader import extract_ground_truth
from .plot import CHANNEL_NAMES
from .runtime_meta import load_runtime_metadata

_INDEX_TO_NAME = CHANNEL_NAMES  # {1-based channel index: electrode name, e.g. 3: "Fz"}
_NAME_TO_INDEX = {name: idx for idx, name in CHANNEL_NAMES.items()}


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
        raw_mne.set_annotations(annotations)
    raw_mne.export(filepath, fmt="edf", add_ch_type=False,
                   physical_range="channelwise", overwrite=True, verbose=False)
    print(f"EDF written: {filepath}")


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
    ground_truth: dict,
    annotations: "mne.Annotations | None" = None,
) -> None:
    """Write the generic, tool-agnostic recording (EEG channels, AUX, Trigger)
    to raw_signals.edf. No pipeline-internal data -- see runtime_meta.py.

    Channels with a known 10-20 electrode (analysis.plot.CHANNEL_NAMES, e.g.
    index 3 -> "Fz") are labelled with that name; others get "EEG_{index}".
    Only real "UDPSource_*" channels get an electrode name -- synthetic
    "SimulatedSource_*" channels always get "EEG_{idx+1}", since load_runtime()
    reconstructs either as "UDPSource_{idx}" on reload regardless.
    """
    time = ground_truth["time"]
    n    = len(time)

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
    start_ts attribute (exact, see load_runtime()) is the authoritative one."""
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


def load_runtime(raw_edf_path: str, meta_h5_path: str) -> tuple[dict, dict, "mne.Annotations"]:
    """Combine raw_signals.edf + runtime_metadata.h5 back into (samples,
    ground_truth, annotations), matching the shapes of
    load_processor_signals()/extract_ground_truth()."""
    meta      = load_runtime_metadata(meta_h5_path)
    recording = load_raw_signals_edf(raw_edf_path)

    fs   = meta["fs"]
    # Authoritative over recording["n"]: EDF pads the final record to a whole second
    n    = meta["n_samples"]
    time = meta["start_ts"] + np.round(np.arange(n) / fs * 1e6).astype(np.int64)

    def source_ts(key):
        return meta["source_ts"].get(key)

    # Only the lowest-indexed recorded channel carries hardware_ts (see write_runtime_metadata)
    hw_ts = meta.get("hardware_ts")
    hw_ts = hw_ts[:n] if hw_ts is not None else None
    first_idx = min(recording["eeg"], default=None)

    raw_prefix = meta["raw_source_class"]
    samples = {}
    for idx, y in recording["eeg"].items():
        samples[f"{raw_prefix}_{idx}"] = {"x": time, "y": y[:n], "source_ts": source_ts(f"{raw_prefix}_{idx}")}
        if idx == first_idx and hw_ts is not None:
            samples[f"{raw_prefix}_{idx}"]["hw_ts"] = hw_ts
    if recording["trigger"] is not None:
        samples["UDPSource_TRIGGER"] = {"x": time, "y": recording["trigger"][:n], "source_ts": source_ts("UDPSource_TRIGGER")}

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

    ground_truth = extract_ground_truth(samples)
    if ground_truth is None:
        raise ValueError(f"{raw_edf_path}: could not reconstruct ground truth (no EEG_* channels found).")

    sim = meta["simulation"]
    for field in ("true_amplitude", "true_phase", "true_inst_freq"):
        if field in sim:
            ground_truth[field] = sim[field]

    return samples, ground_truth, recording["annotations"]


# ---------------------------------------------------------------------------
# analysis.edf
# ---------------------------------------------------------------------------

def write_analysis_edf(
    filepath: str,
    fs: float,
    ground_truth: dict,
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
        whitened: Whether `filtered` was spectrally whitened (see
            compute_hilbert_reference); if so, it's on a different scale than
            Raw's uV and is stored as "stim" instead of "eeg".
    """
    raw  = ground_truth["raw"]
    time = ground_truth["time"]
    n    = len(raw)

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
