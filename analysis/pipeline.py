"""
The offline analysis of one results folder, in two layers that every command
in main.py builds on:

- load_recording(): its graph config and signals (from the cache, or from the
  .bin files, writing the cache).
- analyse_recording(): the offline f0/phase estimation, the errors of the
  online estimates against it, the ideal stimulus train and the pipeline
  latencies.

load_and_analyse() runs both; iter_recordings()/iter_analyses() run them over
many folders.
"""

import traceback
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .cache import Recording, load_signals
from .clas_file_reader import GraphConfig, load_graph_config
from .estimation_errors import compute_errors
from .latency import Latencies, compute_latencies
from .offline_estimation import OfflineEstimate, estimate_offline
from .stimulus import compute_target_stimulus


@dataclass
class RecordingAnalysis:
    """Everything derived from one results folder: the loaded (or cached)
    recording plus the offline f0/Hilbert/error analysis of it."""
    graph:            GraphConfig
    recording:        Recording
    offline:          OfflineEstimate
    errors:           list[dict]
    stim_ref:         np.ndarray | None      # ideal stimulus train (None without StimControl)
    latency:          Latencies | None       # None with fewer than two timestamped processors


def load_recording(results_dir: str) -> "tuple[GraphConfig, Recording] | None":
    """Graph config and signals of one results folder: from the cache if
    available, else from the .bin files (then writing the cache). None if the
    folder can't be loaded."""
    graph = load_graph_config(results_dir)
    if graph is None:
        print(f"Error: no graph config (.yaml) found in {results_dir}")
        return None

    recording = load_signals(results_dir, graph)
    if recording is None:
        return None
    return graph, recording


def analyse_recording(graph: GraphConfig, recording: Recording, f0: float, f0_is_truth: bool = False,
                      whiten: bool = False) -> RecordingAnalysis:
    """Run the offline analysis on a loaded recording: the f0 estimate,
    Hilbert reference, phase/f0/stimulus-edge errors, target stimulus and
    pipeline latencies.

    Args:
        f0: Centre frequency (Hz) for the pre-Hilbert bandpass; used as a
            fallback unless f0_is_truth is set.
        f0_is_truth: Use f0 directly instead of the offline estimate, for
            synthetic runs with a known carrier frequency.
        whiten: Fit the 1/f aperiodic component and use it for spectral
            whitening in estimate_offline().
    """
    fs = graph.fs

    # Offline f0 estimate, Hilbert phase and continuous f0
    offline = estimate_offline(recording.raw, fs, f0, f0_is_truth=f0_is_truth, whiten=whiten)

    # Phase/f0/stimulus-edge errors of the online estimates
    errors = compute_errors(recording, offline.hilbert_phase, offline.f0_continuous, fs, graph)

    # Ideal stimulus train, for visualisation / EDF export
    stim_ref = None
    if "StimControl" in graph.processors:
        stim_ref = compute_target_stimulus(recording, offline.hilbert_phase, offline.f0_continuous, fs, graph)

    # Pipeline and audio onset latencies, from the processors' source timestamps
    latency = compute_latencies(recording, graph)

    return RecordingAnalysis(graph=graph, recording=recording, offline=offline,
                             errors=errors, stim_ref=stim_ref, latency=latency)


def load_and_analyse(f0: float, results_dir: str, f0_is_truth: bool = False,
                     whiten: bool = False) -> "RecordingAnalysis | None":
    """load_recording() + analyse_recording() of one results folder; None if
    it can't be loaded."""
    loaded = load_recording(results_dir)
    if loaded is None:
        return None
    graph, recording = loaded
    return analyse_recording(graph, recording, f0, f0_is_truth=f0_is_truth, whiten=whiten)


def iter_recordings(run_dirs: list[str]) -> Iterator[tuple[str, GraphConfig, Recording]]:
    """(results_dir, graph, recording) of each of run_dirs, see _iter_folders()."""
    for results_dir, (graph, recording) in _iter_folders(run_dirs, load_recording):
        yield results_dir, graph, recording


def iter_analyses(run_dirs: list[str], f0: float, **kwargs) -> Iterator[tuple[str, RecordingAnalysis]]:
    """(results_dir, RecordingAnalysis) of each of run_dirs, see
    load_and_analyse() for f0/kwargs and _iter_folders()."""
    yield from _iter_folders(run_dirs, lambda results_dir: load_and_analyse(f0, results_dir, **kwargs))


def _iter_folders(run_dirs: list[str], load) -> Iterator[tuple[str, object]]:
    """(results_dir, load(results_dir)) for each of run_dirs. Prints progress,
    and skips folders that fail (printing the traceback) or can't be loaded.
    Yields one at a time, so pooled analyses don't keep every recording in memory."""
    for i, results_dir in enumerate(run_dirs, 1):
        print(f"[{i}/{len(run_dirs)}] {results_dir}")
        try:
            result = load(results_dir)
        except Exception:
            print(f"  FAILED: {results_dir}")
            traceback.print_exc()
            continue
        if result is not None:
            yield results_dir, result
