"""Shared filesystem locations for the research/frequency package."""
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../research/frequency/iaf_compare
PROJECT_DIR = PKG_DIR.parent                        # .../research/frequency

# Shared plots folder (<repo>/analysis_output): every .pgf/.pdf plot and every
# .tex table lands here (plot_sweep + plot_reject), so they sit next to each other.
OUTPUTS_DIR = PROJECT_DIR.parent.parent / "analysis_output" / "outputs"
FIGURE_DIR = OUTPUTS_DIR / "figures"                # .pgf/.pdf plots (incl. strategy_roc)
TABLE_DIR = OUTPUTS_DIR                             # .tex tables (conditions, mae summary, reject accuracy/F1)

# Numeric results stay project-local, alongside the HDF5 sweep output.
RESULTS_DIR = PROJECT_DIR / "outputs"
RESULTS_H5 = RESULTS_DIR / "iaf_results.h5"
REJECT_DIR = RESULTS_DIR / "reject"                 # strategy_comparison_*.csv


def ensure_output_dirs():
    """Create the output trees (shared plots + project-local results) if they
    do not exist yet."""
    for d in (OUTPUTS_DIR, FIGURE_DIR, TABLE_DIR, RESULTS_DIR, REJECT_DIR):
        d.mkdir(parents=True, exist_ok=True)
