"""
EEG pipeline analysis: exports signals to EDF and plots phase/frequency errors.

Usage:
    python plot_results.py

Outputs:
    pipeline_signals.edf   — all signals (raw, filtered, phase estimates, etc.)
    error_analysis.png     — error time series + histograms only
"""

import os
import yaml
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import mne
import tkinter as tk
from tkinter import filedialog

from postprocess_results import write_edf, extract_ground_truth, get_erp_windows, load_processor_signals, plot_erp_latency

# Automatic link to the last run results
RESULTS_DIR = "_last_run"

COMMON_BBOX = dict(
    facecolor="white",
    edgecolor="0.8",
    boxstyle="round,pad=0.2",
    alpha=0.85,
)

def pick_folders():
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
    tk.Button(win, text="Remove", command=remove).pack(side=tk.LEFT)
    tk.Button(win, text="OK", command=win.destroy).pack(side=tk.RIGHT, padx=5)
    win.mainloop()
    return selected


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_results(
) -> None:

    windows = []

    folders = pick_folders()

    for results_dir in folders:

        graph_file = glob.glob(os.path.join(results_dir, "*.yaml"))
        if Path(results_dir).is_symlink():
            edf_path = os.path.join(results_dir, Path(results_dir).readlink().stem + ".edf")
        else:
            edf_path = os.path.join(results_dir, Path(results_dir).stem + ".edf")

        with open(graph_file[0], "r") as f:
            graph_config = yaml.safe_load(f)

        fs = graph_config.get("graph", {}).get("defaults", {}).get("fs", None)

        processors = graph_config.get("graph", {}).get("processors", [])
        if processors is None:
            raise ValueError(f"No processors found in graph config.")

        # 1. Load raw processor outputs
        samples = load_processor_signals(fs, results_dir, processors)

        if os.path.basename(graph_file[0]) == "ERPCLAS.yaml":
            ground_truth = extract_ground_truth(samples)
            write_edf(edf_path, fs, ground_truth, samples)
            windows.extend(get_erp_windows(fs, samples, channel=1))

    plot_erp_latency(windows, fs)
    
    plt.show()




if __name__ == "__main__":

    analyse_results()

    # _last_run
    # results/eike_20260529_1600