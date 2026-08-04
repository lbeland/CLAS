"""
Main analysis entry point. Run with:
    python -m analysis.main          (from CLAS/)
    python postprocess_results.py    (via the root launcher)
"""

import glob
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.loader import load_processor_signals, analyse_latencies, extract_ground_truth
from analysis.iaf    import estimate_iaf, estimate_iaf_with_phase
from analysis.core   import compute_hilbert_reference, compute_errors
from analysis.plot   import write_edf, plot_errors, plot_spectrum, plot_time_series, \
                            plot_iaf, get_erp_windows, plot_erp_latency


def analyse_results(f0: float, results_dir: str) -> None:
    graph_files = glob.glob(os.path.join(results_dir, "*.yaml"))
    if not graph_files:
        print(f"Error: no graph config (.yaml) found in {results_dir}")
        return

    graph_file = graph_files[0]
    edf_stem   = Path(results_dir).readlink().stem if Path(results_dir).is_symlink() \
                 else Path(results_dir).stem
    edf_path   = os.path.join(results_dir, edf_stem + ".edf")

    with open(graph_file) as f:
        graph_config = yaml.safe_load(f)

    fs         = graph_config.get("graph", {}).get("defaults", {}).get("fs")
    processors = graph_config.get("graph", {}).get("processors", [])
    if processors is None:
        raise ValueError("No processors found in graph config.")

    # 1. Load processor outputs
    samples = load_processor_signals(fs, results_dir, processors)

    # 2. Identify ground-truth signals
    ground_truth = extract_ground_truth(samples)
    if ground_truth is None:
        print("Error: no source signal found (SourceClient or Producer). Aborting.")
        return

    # 3. Pipeline latency analysis
    analyse_latencies(samples, ground_truth, graph_config)

    # 4. ERPCLAS-specific: ERP average plot
    if os.path.basename(graph_file) == "ERPCLAS.yaml":
        windows = get_erp_windows(fs, samples, channel=1)
        plot_erp_latency(windows, fs)

    # 5. Estimate IAF offline on the full recording
    iaf, aperiodic_params = estimate_iaf(ground_truth["raw"], fs)
    if np.all(~np.isfinite(iaf)):
        print("Could not estimate IAF from ground truth; using default f0 =", f0)
    else:
        f0 = float(np.nanmean(iaf))
        print(f"Estimated IAFs on 30-second windows: mean={f0:.2f} Hz, all={iaf}")

    # 6. Offline Hilbert reference
    raw = ground_truth["raw"]
    # filtered2, hilbert_phase2 = compute_hilbert_reference(raw, fs, f0, aperiodic_params=aperiodic_params)
    filtered, hilbert_phase = compute_hilbert_reference(raw, fs, f0)  # without aperiodic params

    iaf_continuous = estimate_iaf_with_phase(hilbert_phase, fs, f0)

    # 7. Compute errors
    start_ts = ground_truth["time"][0]
    errors, stim_ref = compute_errors(samples, ground_truth, hilbert_phase, start_ts, fs, graph_config)
    # errors2, stim_ref2 = compute_errors(samples, ground_truth, hilbert_phase2, start_ts, fs, graph_config)

    # 8. Export to EDF
    write_edf(edf_path, fs, ground_truth, samples, hilbert_phase, stim_ref, filtered)

    # 9. Plot errors
    plot_errors(errors)
    # plot_errors(errors2, title="Without aperiodic params")

    # 10. IAF time series
    plot_iaf(ground_truth, iaf_continuous, samples, start_ts)

    # 11. Plot spectrum
    # plot_spectrum(raw, samples, fs)

    # 12. Plot time series
    time_range = None #(10, 20)  # set to e.g. (17, 18) to zoom in seconds
    # plot_time_series(ground_truth, samples, hilbert_phase, stim_ref, time_range=time_range)

    plt.show()


if __name__ == "__main__":
    analyse_results(f0=10, results_dir="_last_run")
