"""Shared filesystem locations for the compare_f0_algos package.

Importing this module also puts the vendored SIMparam ``code/`` directory on
``sys.path`` so ``from sims import gen_power_vals_fn`` works from anywhere in
the package (SIMparam is a plain nested clone, not an installable package).
"""
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../compare_f0_algos/iaf_compare
PROJECT_DIR = PKG_DIR.parent                        # .../compare_f0_algos

OUTPUTS_DIR = Path("/home/linda/Documents/MA/plots") / "outputs"
FIGURE_DIR = OUTPUTS_DIR / "figures"                # plot_sweep .pgf/.pdf output
TABLE_DIR = OUTPUTS_DIR                             # conditions_table.tex
REJECT_DIR = OUTPUTS_DIR / "reject"                 # strategy_comparison_*.csv, strategy_roc.png

RESULTS_H5 = PROJECT_DIR / "outputs" / "iaf_results.h5"

SIMPARAM_CODE = PROJECT_DIR / "SIMparam" / "code"
if SIMPARAM_CODE.is_dir() and str(SIMPARAM_CODE) not in sys.path:
    sys.path.insert(0, str(SIMPARAM_CODE))


def ensure_output_dirs():
    """Create the outputs/ tree if it does not exist yet."""
    for d in (OUTPUTS_DIR, FIGURE_DIR, REJECT_DIR):
        d.mkdir(parents=True, exist_ok=True)
