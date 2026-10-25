"""
ERP epoching, plus two multi-folder ERP analyses: one ERP curve per subject
(analyse_erp_by_subject), and the pooled average of hand-picked runs
(analyse_erp, with a folder-picker dialog).

Run with:
    python -m analysis.main erp   (from CLAS/, one curve per subject)
    python -m analysis.erp        (from CLAS/, folder-picker dialog)
"""

import os

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

from analysis.clas_file_reader import load_graph_config, find_session_run_dirs, CHANNEL_NAMES, EOG_CHANNEL_NAMES
from analysis.cache            import load_signals
from analysis.pipeline         import iter_recordings
from analysis.stimulus         import get_edges, get_trigger_binary
from analysis.plot             import plot_erp_latency, plot_erp_by_subject


def get_erp_windows(fs: float, samples: dict, channel: list[int] = [1],
                     bandpass: tuple[float, float] = (2.0, 30.0),
                     reject_uv: float = 10000.0,
                     average: bool = False) -> list[dict]:
    """Extract EEG epochs around each trigger onset (-250 ms to +500 ms) for each channel.

    The EEG is bandpass-filtered (default 2-30 Hz) before epoching, and any
    epoch whose peak amplitude exceeds ``reject_uv`` (default +/-100 uV) is
    discarded. Each returned window carries a 1-based "channel" key so
    callers can group windows by channel.

    If ``average`` is True, each channel's per-trial epochs are collapsed
    into a single mean window (carrying an added "n_epochs" key) before
    returning, instead of one window per trial -- much lighter to keep
    around when only the averaged ERP is needed (e.g. one curve per subject
    pooled across many recordings).
    """
    all_windows    = []

    for ch in channel:
        ch0 = ch - 1  # zero-based

        # if samples.get("ecHTFilter") is not None:
        #     eeg_y, eeg_t = samples["ecHTFilter"]["y"], samples["ecHTFilter"]["x"]
        if samples.get(f"UDPSource_{ch0}") is not None:
            eeg_y, eeg_t = samples[f"UDPSource_{ch0}"]["y"], samples[f"UDPSource_{ch0}"]["x"]
        elif samples.get(f"SimulatedSource_{ch0}") is not None:
            eeg_y, eeg_t = samples[f"SimulatedSource_{ch0}"]["y"], samples[f"SimulatedSource_{ch0}"]["x"]
        else:
            print(f"Error: no EEG signal found for channel {ch}.")
            continue

        trigger_binary = get_trigger_binary(samples)
        if trigger_binary is None:
            print("Error: no trigger signal found.")
            return []

        time_s         = (eeg_t - eeg_t[0]) / 1e6
        onsets         = get_edges(trigger_binary, "rising")
        offsets        = get_edges(trigger_binary, "falling")

        # Bandpass-filter the continuous EEG before epoching, so filter edge
        # artifacts don't contaminate the epoch boundaries.
        sos   = butter(1, bandpass, btype="band", fs=fs, output="sos")
        eeg_y = sosfiltfilt(sos, eeg_y)

        ch_windows = []
        n_rejected = 0
        for idx in onsets:
            start_idx = idx - int(0.25 * fs)
            end_idx   = idx + int(0.5  * fs)
            if start_idx >= 0 and end_idx < len(eeg_y):
                epoch = eeg_y[start_idx:end_idx]
                if np.max(np.abs(epoch)) > reject_uv:
                    n_rejected += 1
                    continue
                later_offsets = offsets[offsets > idx]
                stim_dur_s    = (time_s[later_offsets[0]] - time_s[idx]) if len(later_offsets) else None
                ch_windows.append({"channel":    ch,
                                   "time_s":     time_s[start_idx:end_idx],
                                   "eeg_y":       epoch,
                                   "stim_dur_s":  stim_dur_s})

        if n_rejected:
            print(f"Channel {ch}: rejected {n_rejected} epoch(s) with peak amplitude > +/-{reject_uv:.0f} uV.")

        if average and ch_windows:
            stim_durs = [w["stim_dur_s"] for w in ch_windows if w["stim_dur_s"] is not None]
            all_windows.append({"channel":   ch,
                                "time_s":     ch_windows[0]["time_s"],
                                "eeg_y":      np.mean([w["eeg_y"] for w in ch_windows], axis=0),
                                "stim_dur_s": float(np.mean(stim_durs)) if stim_durs else None,
                                "n_epochs":   len(ch_windows)})
        else:
            all_windows.extend(ch_windows)

    return all_windows


def analyse_erp() -> None:
    """Load ERP windows from multiple result folders and plot the pooled average."""
    windows = []
    fs      = None

    for results_dir in pick_folders():
        try:
            graph = load_graph_config(results_dir)
        except ValueError as e:
            print(f"Warning: {e} Skipping.")
            continue
        if graph is None:
            print(f"Warning: no .yaml found in {results_dir}, skipping.")
            continue
        fs = graph.fs

        is_erp = os.path.basename(graph.path) == "ERPCLAS.yaml"
        recording = load_signals(results_dir, graph, write_cache=is_erp)
        if recording is None:
            continue

        if is_erp:
            windows.extend(get_erp_windows(fs, recording.samples, channel=[1]))

    if windows and fs is not None:
        plot_erp_latency(windows, fs)
        plt.show()
    else:
        print("No ERP windows collected.")


def analyse_erp_by_subject(base_dir: str = "results", session_glob: str = "CLAS_*",
                           channel: int = 3) -> None:
    """Plot the ERP (default channel 3 = Fz) as one curve per subject, all
    centred on stimulus onset, so per-subject ERP shape/latency can be
    compared at a glance. Subjects are discovered the same way as
    analyse_pooled_errors() (every results/CLAS_* session folder); within
    each, every ERPCLAS.yaml run (normally just the "*_erp_meas_*" one) is
    pooled into that subject's curve."""
    grouped: dict[str, list[str]] = {}
    for run_dir in find_session_run_dirs(base_dir, session_glob, graph_name="ERPCLAS.yaml"):
        grouped.setdefault(os.path.dirname(run_dir), []).append(run_dir)
    print(f"Found {len(grouped)} subject(s) with an ERP measurement under {base_dir}/{session_glob}/*.")

    subject_curves = []
    for session_dir, run_dirs in grouped.items():
        print(f"[{os.path.basename(session_dir)}]")
        windows = []
        fs = None
        for _, graph, recording in iter_recordings(run_dirs):
            fs = graph.fs
            windows += get_erp_windows(fs, recording.samples, channel=[channel], average=True)
        if windows:
            subject_curves.append({"name": os.path.basename(session_dir), "windows": windows, "fs": fs})

    if not subject_curves:
        print("No ERP windows collected; nothing to plot.")
        return

    plot_erp_by_subject(subject_curves)
    plt.show()


def pick_folders() -> list[str]:
    """Open a small Tk dialog to select one or more result folders."""
    import tkinter as tk  # lazy: only this dialog needs Tk
    from tkinter import filedialog

    selected = []

    def add():
        f = filedialog.askdirectory()
        if f and f not in selected:
            selected.append(f)
            listbox.insert(tk.END, f)

    def remove():
        for i in reversed(listbox.curselection()):
            listbox.delete(i)
            selected.pop(i)

    win = tk.Tk()
    win.title("Select Folders")
    listbox = tk.Listbox(win, width=60, selectmode=tk.MULTIPLE)
    listbox.pack(padx=10, pady=10)
    tk.Button(win, text="Add Folder", command=add).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(win, text="Remove",     command=remove).pack(side=tk.LEFT)
    tk.Button(win, text="OK",         command=win.destroy).pack(side=tk.RIGHT, padx=5)
    win.mainloop()
    return selected

if __name__ == "__main__":
    analyse_erp()
