"""
Main analysis entry point. Run with:
    python -m analysis.main <command>   (from CLAS/)

Commands (see --help of each for its flags):
    single    full analysis + plots for one results folder
    all       analyse every results folder (batch smoke test / cache building)
    pooled    pool errors across every session's recordings (pooled.py)
    erp       one ERP curve per subject (erp.py)
    spectrum  Stim On vs Stim Off power change per condition (spectrum_analysis.py)
    sweep     SNR / noise-colour sweep error table (sweep.py)
"""

import argparse
import os
import traceback

import matplotlib.pyplot as plt

from analysis.clas_file_reader  import find_result_dirs
from analysis.pipeline          import RecordingAnalysis, load_recording, analyse_recording
from analysis.results_io            import write_analysis_edf
from analysis.erp               import get_erp_windows, analyse_erp_by_subject
from analysis.plot              import plot_errors, plot_spectrum, plot_stack, plot_erp_latency, plot_latencies
from analysis.pooled            import analyse_pooled_errors
from analysis.spectrum_analysis import analyse_all as analyse_spectrum, TF_COMPARISONS
from analysis.stimulus          import STIM_ONSET_DEGS
from analysis.sweep             import snr_sweep_table
from analysis import report


def analyse_results(results_dir: str, f0: float=10.0, plot: bool = True,
                    f0_is_truth: bool = False, whiten: bool = False, full_analysis: bool = True) -> None:
    """Analyse one results folder: print the report, write analysis.edf and
    plot the latency, error distributions, spectrum and time series. With
    full_analysis=False, only load it (building the cache)."""
    report.print_header(results_dir)
    loaded = load_recording(results_dir)
    if loaded is None:
        return
    graph, recording = loaded
    report.print_data_line(graph, recording)

    if full_analysis:
        analysis = analyse_recording(graph, recording, f0, f0_is_truth=f0_is_truth, whiten=whiten)
        report.print_recording_summary(analysis)
        _export_analysis_edf(analysis, results_dir)
        if plot:
            _plot_recording(analysis)
            plt.show()


def _export_analysis_edf(analysis: RecordingAnalysis, results_dir: str) -> None:
    """Write the analysis outputs to results_dir/analysis.edf (cheap; rebuilt
    every run from the cached runtime signals)."""
    recording, offline = analysis.recording, analysis.offline
    write_analysis_edf(os.path.join(results_dir, "analysis.edf"), analysis.graph.fs,
                       recording.raw, recording.time, recording.samples,
                       offline.filtered, offline.hilbert_phase, offline.f0_continuous,
                       analysis.stim_ref, annotations=recording.annotations,
                       whitened=offline.X_white is not None)


def _plot_recording(analysis: RecordingAnalysis) -> None:
    """Latency, ERP (ERPCLAS runs only), error distribution, spectrum and time-series figures."""
    recording, offline, fs = analysis.recording, analysis.offline, analysis.graph.fs

    # Pipeline latency. Reproducible from cache too, since runtime_metadata.h5
    # preserves source_ts exactly (unlike an EDF round-trip).
    if analysis.latency is not None:
        plot_latencies(analysis.latency)

    # ERPCLAS-specific: ERP average plot
    if os.path.basename(analysis.graph.path) == "ERPCLAS.yaml":
        windows = get_erp_windows(fs, recording.samples, channel=[4,3,27,26])
        plot_erp_latency(windows, fs)

    # Polar error distributions
    plot_errors(analysis.errors)

    # Plot spectrum
    plot_spectrum(recording.raw, recording.samples, offline.filtered, fs,
                  offline.aperiodic_params, offline.X_white)

    # Time-domain panels: stack any subset of "f0"/"phase"/"phase_error"/"signals"
    # e.g. plot_stack(analysis, ["signals", "phase_error"], save_as="sig_vs_err")
    time_range = None # (20.5, 23.5)  # e.g. (17, 18) to zoom in on a few seconds
    # plot_stack(analysis, ["signals", "f0", "phase", "phase_error"], time_range=time_range,
    #            save_as="time_series")
    plot_stack(analysis, ["f0"], time_range=time_range,
            save_as="time_series")

def analyse_all_results(f0: float = 10, base_dir: str = "results", full_analysis: bool = True) -> None:
    """Batch smoke-test: run analyse_results() non-interactively (no
    plot) over every results subfolder under base_dir, keeping going
    past per-folder errors so one bad run doesn't block the rest. Prints a
    full traceback for each failure plus a pass/fail summary at the end."""
    run_dirs = find_result_dirs(base_dir)
    print(f"Found {len(run_dirs)} results folder(s) under {base_dir!r}.")

    failures = []
    for i, results_dir in enumerate(run_dirs, 1):
        print(f"\n{'=' * 80}\n[{i}/{len(run_dirs)}] {results_dir}\n{'=' * 80}")
        try:
            analyse_results(results_dir, f0, plot=False, full_analysis=full_analysis)
        except Exception:
            print(f"FAILED: {results_dir}")
            traceback.print_exc()
            failures.append(results_dir)

    print(f"\n{'=' * 80}\n{len(run_dirs) - len(failures)}/{len(run_dirs)} succeeded")
    if failures:
        print("Failed folders:")
        for results_dir in failures:
            print(f"  {results_dir}")
    print("=" * 80)


def _parse_pair(s: str) -> tuple[int, int]:
    a, b = s.split(",")
    return int(a), int(b)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline CLAS analysis entry point")
    subparsers = parser.add_subparsers(dest="command", required=True)

    all_p = subparsers.add_parser(
        "all", help="analyse_all_results(): analyze every results/ subfolder to generate .edf and .h5 files.")
    all_p.add_argument("--base-dir", default="results",
                        help="Base directory to search for results folders (default: results).")
    all_p.add_argument("--f0", type=float, default=10.0,
                        help="Fallback f0 (Hz) if offline estimation fails (default: 10.0).")
    all_p.add_argument("--no-full-analysis", dest="full_analysis", action="store_false",
                        help="Skip f0/Hilbert/error analysis; just load (and cache) each folder.")

    single_p = subparsers.add_parser(
        "single", help="analyse_results(): full analysis + plots for one results folder.")
    single_p.add_argument("results_dir", nargs="?", default="_last_run",
                           help="Results folder to analyse (default: _last_run).")
    single_p.add_argument("--f0", type=float, default=10.0,
                           help="Centre frequency (Hz) for the pre-Hilbert bandpass; fallback "
                                "unless --f0-is-truth is set (default: 10.0).")
    single_p.add_argument("--f0-is-truth", action="store_true",
                           help="Use --f0 directly instead of re-estimating it offline.")
    single_p.add_argument("--whiten", action="store_true",
                           help="Apply 1/f spectral whitening before the offline bandpass.")
    single_p.add_argument("--no-plot", dest="plot", action="store_false",
                           help="Don't create any plots (just load and analyse the data).")
    single_p.add_argument("--no-full-analysis", dest="full_analysis", action="store_false",
                           help="Skip f0/Hilbert/error analysis and plotting; just load the folder.")

    pooled_p = subparsers.add_parser(
        "pooled", help="analyse_pooled_errors(): pool errors across every session's recordings.")
    pooled_p.add_argument("--f0", type=float, default=10.0)
    pooled_p.add_argument("--base-dir", default="results")
    pooled_p.add_argument("--session-glob", default="CLAS_*",
                           help="Glob (under --base-dir) matching session folders (default: CLAS_*).")
    pooled_p.add_argument("--names", nargs="*", default=None,
                           help="Only include recordings whose path contains one of these "
                                "substrings (case-insensitive), e.g. --names victor.")

    erp_p = subparsers.add_parser(
        "erp", help="analyse_erp_by_subject(): plot one ERP curve per subject.")
    erp_p.add_argument("--base-dir", default="results")
    erp_p.add_argument("--session-glob", default="CLAS_*")
    erp_p.add_argument("--channel", type=int, default=3,
                        help="Channel index to average for the ERP (default: 3 = Fz).")

    spectrum_p = subparsers.add_parser(
        "spectrum", help="spectrum_analysis.analyse_all(): log power change spectrum, opposite-phase "
                         "time-frequency t-maps, and pooled Stim On/Off power change, per "
                         "stim_onset_deg condition.")
    spectrum_p.add_argument("--base-dir", default="results",
                             help="Base directory to search for session folders (default: results).")
    spectrum_p.add_argument("--channel", default="Fz",
                             help="Electrode to analyse (default: Fz).")
    spectrum_p.add_argument("--stim-onset-degs", type=int, nargs="*", default=list(STIM_ONSET_DEGS),
                             help=f"stim_onset_deg conditions to include (default: {list(STIM_ONSET_DEGS)}).")
    spectrum_p.add_argument("--comparisons", type=_parse_pair, nargs="*", default=list(TF_COMPARISONS),
                             help="Opposite-phase condition pairs for the time-frequency t-maps, as "
                                  "'a,b' (default: %(default)s). Pass with no values to skip the t-maps.")
    spectrum_p.add_argument("--win-s", type=float, default=1.0, help="Welch window length (s) (default: 1.0).")
    spectrum_p.add_argument("--step-s", type=float, default=1.0, help="Welch hop between windows (s) (default: 1.0).")
    spectrum_p.add_argument("--smooth-window-s", type=float, default=10.0,
                             help="Time-resolved map smoothing window (s) (default: 10.0).")
    spectrum_p.add_argument("--no-csd", dest="use_csd", action="store_false",
                             help="Skip the surface Laplacian / CSD re-referencing step.")

    sweep_p = subparsers.add_parser(
        "sweep", help="snr_sweep_table(): error table of an SNR / noise-colour sweep (see sweep_clas.py).")
    sweep_p.add_argument("sweep_dir", nargs="?", default="results/snr_sweep",
                          help="Folder containing the sweep's run folders (default: results/snr_sweep).")
    sweep_p.add_argument("--f0", type=float, default=10.0,
                          help="Bandpass centre frequency (Hz) when --no-f0-is-truth is set (default: 10.0).")
    sweep_p.add_argument("--no-f0-is-truth", dest="f0_is_truth", action="store_false",
                          help="Re-estimate f0 offline instead of using each run's known carrier frequency.")
    sweep_p.add_argument("--whiten", action="store_true",
                          help="Apply 1/f spectral whitening before the offline bandpass.")
    sweep_p.add_argument("--out-dir", default=None,
                          help="Output folder for the .csv/.tex table (default: analysis_output).")
    sweep_p.add_argument("--label", default=None,
                          help="LaTeX \\label of the table (default: tab:snr_sweep_echt).")

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.command == "all":
        analyse_all_results(f0=args.f0, base_dir=args.base_dir, full_analysis=args.full_analysis)
    elif args.command == "single":
        analyse_results(results_dir=args.results_dir, f0=args.f0, plot=args.plot,
                         f0_is_truth=args.f0_is_truth, whiten=args.whiten,
                         full_analysis=args.full_analysis)
    elif args.command == "pooled":
        analyse_pooled_errors(f0=args.f0, names=args.names, base_dir=args.base_dir,
                              session_glob=args.session_glob)
    elif args.command == "erp":
        analyse_erp_by_subject(base_dir=args.base_dir, session_glob=args.session_glob,
                               channel=args.channel)
    elif args.command == "spectrum":
        analyse_spectrum(base_dir=args.base_dir, stim_onset_degs=tuple(args.stim_onset_degs),
                         comparisons=tuple(args.comparisons), channel=args.channel,
                         win_s=args.win_s, step_s=args.step_s, smooth_window_s=args.smooth_window_s,
                         use_csd=args.use_csd)
    elif args.command == "sweep":
        snr_sweep_table(args.sweep_dir, f0=args.f0, out_dir=args.out_dir,
                        f0_is_truth=args.f0_is_truth, whiten=args.whiten, label=args.label)
