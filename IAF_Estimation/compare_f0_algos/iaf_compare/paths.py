"""Shared filesystem locations for the compare_f0_algos package.

Importing this module also puts the vendored SIMparam ``code/`` directory on
``sys.path`` so ``from sims import gen_power_vals_fn`` works from anywhere in
the package (SIMparam is a plain nested clone, not an installable package).
"""
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../compare_f0_algos/iaf_compare
PROJECT_DIR = PKG_DIR.parent                        # .../compare_f0_algos

# Shared thesis plots folder: every .pgf/.pdf plot and every .tex table lands
# here (plot_sweep + plot_reject), so they sit next to each other.
OUTPUTS_DIR = Path("/home/linda/Documents/MA/plots") / "outputs"
FIGURE_DIR = OUTPUTS_DIR / "figures"                # .pgf/.pdf plots (incl. strategy_roc)
TABLE_DIR = OUTPUTS_DIR                             # .tex tables (conditions, mae summary, reject accuracy/F1)

# Numeric results stay project-local, alongside the HDF5 sweep output.
RESULTS_DIR = PROJECT_DIR / "outputs"
RESULTS_H5 = RESULTS_DIR / "iaf_results.h5"
REJECT_DIR = RESULTS_DIR / "reject"                 # strategy_comparison_*.csv

SIMPARAM_CODE = PROJECT_DIR / "SIMparam" / "code"
if SIMPARAM_CODE.is_dir() and str(SIMPARAM_CODE) not in sys.path:
    sys.path.insert(0, str(SIMPARAM_CODE))


def ensure_output_dirs():
    """Create the output trees (shared plots + project-local results) if they
    do not exist yet."""
    for d in (OUTPUTS_DIR, FIGURE_DIR, TABLE_DIR, RESULTS_DIR, REJECT_DIR):
        d.mkdir(parents=True, exist_ok=True)
