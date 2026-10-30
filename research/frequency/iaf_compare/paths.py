"""Shared filesystem locations for the research/frequency package."""
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent   # .../research/frequency/iaf_compare
PROJECT_DIR = PKG_DIR.parent     # .../research/frequency

# Numeric results stay project-local, alongside the HDF5 sweep output.
OUTPUTS_DIR = PROJECT_DIR / "outputs"
RESULTS_H5 = OUTPUTS_DIR / "iaf_results.h5"

FIGURE_DIR = OUTPUTS_DIR / "figures"   # .pgf/.pdf plots (incl. strategy_roc)
TABLE_DIR = OUTPUTS_DIR        # .tex tables (conditions, mae summary, reject accuracy/F1)

REJECT_DIR = OUTPUTS_DIR / "reject"    # strategy_comparison_*.csv


def ensure_output_dirs():
    """Create the output trees (shared plots + project-local results) if they
    do not exist yet."""
    for d in (FIGURE_DIR, TABLE_DIR, OUTPUTS_DIR, REJECT_DIR):
        d.mkdir(parents=True, exist_ok=True)
