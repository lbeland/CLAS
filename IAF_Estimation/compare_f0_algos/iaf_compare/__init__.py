"""Peak-frequency (individual alpha frequency) algorithm comparison on
synthetic FOOOF-model signals.

Entry-point scripts live one level up: ``run_sweep.py`` (generate the sweep +
write ``outputs/iaf_results.h5``), ``plot_sweep.py`` (figures from it),
``reject_compare.py`` / ``plot_roc.py`` (peak-rejection strategy comparison).
"""
from . import paths, config, signal_gen, spectra, algorithms, metrics, pipeline, io_hdf5

from .config import FIXED, DEFAULT, SWEEPS, WINDOW_LENGTHS_SEC, N_SEEDS, save_conditions_latex_table
from .signal_gen import generate_signal
from .spectra import precompute_dpss, _init_dpss_cache
from .algorithms import run_algorithms
from .metrics import compute_pooled_metrics
from .pipeline import (
    build_sweep, process_condition, process_window_length_condition, run_window_analysis,
)
from .io_hdf5 import write_results, load_metrics, load_samples_for_plot, filter_df

__all__ = [
    "paths", "config", "signal_gen", "spectra", "algorithms", "metrics",
    "pipeline", "io_hdf5",
    "FIXED", "DEFAULT", "SWEEPS", "WINDOW_LENGTHS_SEC", "N_SEEDS",
    "save_conditions_latex_table", "generate_signal", "precompute_dpss",
    "_init_dpss_cache", "run_algorithms", "compute_pooled_metrics",
    "build_sweep", "process_condition", "process_window_length_condition",
    "run_window_analysis", "write_results", "load_metrics", "load_samples_for_plot",
    "filter_df",
]
