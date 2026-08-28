"""
Multi-folder ERP analysis: pool ERP windows across several runs and plot the average.

Run with:
    python calc_erp.py          (via the root launcher)
    python -m analysis.erp      (from CLAS/)
"""

import os
import sys
import glob
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.loader       import load_processor_signals, extract_ground_truth
from analysis.edf_io       import write_raw_signals_edf, load_runtime
from analysis.runtime_meta import write_runtime_metadata
from analysis.plot         import get_erp_windows, plot_erp_latency


def pick_folders() -> list[str]:
    """Open a small Tk dialog to select one or more result folders."""
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


def analyse_erp() -> None:
    """Load ERP windows from multiple result folders and plot the pooled average."""
    windows = []
    fs      = None

    for results_dir in pick_folders():
        graph_files = glob.glob(os.path.join(results_dir, "*.yaml"))
        if not graph_files:
            print(f"Warning: no .yaml found in {results_dir}, skipping.")
            continue

        with open(graph_files[0]) as f:
            graph_config = yaml.safe_load(f)

        fs         = graph_config.get("graph", {}).get("defaults", {}).get("fs")
        processors = graph_config.get("graph", {}).get("processors", [])
        if processors is None:
            print(f"Warning: no processors in {graph_files[0]}, skipping.")
            continue

        raw_edf_path = os.path.join(results_dir, "raw_signals.edf")
        meta_h5_path = os.path.join(results_dir, "runtime_metadata.h5")
        if os.path.exists(raw_edf_path) and os.path.exists(meta_h5_path):
            samples, ground_truth, _ = load_runtime(raw_edf_path, meta_h5_path)
        else:
            samples      = load_processor_signals(fs, results_dir, processors)
            ground_truth = extract_ground_truth(samples)
            if os.path.basename(graph_files[0]) == "ERPCLAS.yaml":
                write_raw_signals_edf(raw_edf_path, fs, samples, ground_truth)
                write_runtime_metadata(meta_h5_path, fs, samples, ground_truth)

        if os.path.basename(graph_files[0]) == "ERPCLAS.yaml":
            windows.extend(get_erp_windows(fs, samples, channel=[1]))

    if windows and fs is not None:
        plot_erp_latency(windows, fs)
        plt.show()
    else:
        print("No ERP windows collected.")


if __name__ == "__main__":
    analyse_erp()
