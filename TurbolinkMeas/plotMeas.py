"""
Analyze pdv_<fs>.csv files produced by measurement.cpp.

Each pdv_<fs>.csv (fs = nominal sample rate, embedded in the filename)
holds only the last PDV_WINDOW_S seconds of the run (see measurement.cpp),
with a header comment line giving the sample rate estimated from the *whole*
run:
    # true_fs_hz=<value>
    sample_counter,pdv_us,inter_sample_us

This script owns all the statistics (measurement.cpp just dumps the raw
buffered Packet Delay Variation, PDV): for every pdv_*.csv found, it
  1. plots PDV over time in its own figure, saved as .pdf and .pgf under
     /home/linda/Documents/MA/plots,
  2. plots a combined PDV histogram across all sample rates, and
  3. writes one combined LaTeX table (pdv_stats.tex) with one row per
     nominal sample rate: estimated true fs, PDV mean/std/max.

Usage:
    python3 plotMeas.py [glob]

`glob` defaults to "pdv_*.csv" in this script's directory.
"""

import glob
import os
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})

ROOT = sys.path[0]  # directory of this script
PLOTS_DIR = "/home/linda/Documents/MA/plots"

TEXTWIDTH = 6.30045
ASPECT_RATIO = 9 / 16

TRUE_FS_RE = re.compile(r"true_fs_hz=([\d.eE+-]+)")
FILENAME_RE = re.compile(r"pdv_(\d+)\.csv$")


def load_pdv(path):
    """Returns (nominal_fs, true_fs, sample_counter, pdv_us, inter_sample_us)."""
    with open(path) as f:
        first_line = f.readline()
    match = TRUE_FS_RE.search(first_line)
    if not match:
        raise ValueError(f"'{path}' is missing the '# true_fs_hz=...' header line")
    true_fs = float(match.group(1))

    fname_match = FILENAME_RE.search(os.path.basename(path))
    if not fname_match:
        raise ValueError(f"Could not infer nominal sample rate from filename '{path}'")
    nominal_fs = float(fname_match.group(1))

    # skip_header=1 to skip past the "# true_fs_hz=..." line explicitly:
    # genfromtxt's own comment-stripping turns it into an empty line rather
    # than skipping it, which throws off names=True's header-row detection.
    data = np.genfromtxt(path, delimiter=",", names=True, skip_header=1)
    return (nominal_fs, true_fs,
            data["sample_counter"], data["pdv_us"], data["inter_sample_us"])


def plot_pdv_over_time(nominal_fs, true_fs, sample_counter, pdv_us):
    width = TEXTWIDTH
    height = width * ASPECT_RATIO
    fig, ax = plt.subplots(figsize=(width, height))

    t = (sample_counter - sample_counter[0]) / true_fs
    ax.plot(t, pdv_us, 'o',markersize=0.5,alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Packet delay variation ($\mu$s)")
    ax.grid(True, linewidth=0.3)
    ax.set_xlim(0,0.5)
    fig.tight_layout()

    stem = f"pdv_{int(nominal_fs)}"
    fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.pgf"))
    fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.pdf"))
    plt.close(fig)


def plot_pdv_histogram(pdv_by_fs, bins=100):
    """pdv_by_fs: list of (nominal_fs, pdv_us) tuples, one per sample rate.

    Renders one subplot per sample rate in a 2x2 grid, sharing a common
    x and y scale so the spreads are directly comparable across rates.
    """
    width = TEXTWIDTH
    height = width * 3 / 4
    fig, axes = plt.subplots(2, 2, figsize=(width, height),
                              sharex=True, sharey=True)

    all_pdv = np.concatenate([pdv_us for _, pdv_us in pdv_by_fs])
    shared_bins = np.linspace(all_pdv.min(), all_pdv.max(), bins + 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (nominal_fs, pdv_us), color in zip(axes.flat, pdv_by_fs, colors):
        ax.hist(pdv_us, bins=shared_bins, density=True, color=color,
                label=f"{int(nominal_fs)} Hz")
        ax.grid(True, linewidth=0.3)

    for ax in axes.flat[len(pdv_by_fs):]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Packet delay variation ($\mu$s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")

    handles, labels = [], []
    for ax in axes.flat[:len(pdv_by_fs)]:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="lower center", ncol=len(pdv_by_fs),
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    stem = "pdv_hist_all"
    fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.pgf"))
    fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.pdf"))
    plt.close(fig)


def pdv_stats(pdv_us):
    stats = {"mean": float(pdv_us.mean()), "std": float(pdv_us.std()), "max": float(pdv_us.max())}
    return stats


def save_latex_table(rows, out_path):
    """rows: list of dicts with nominal_fs, true_fs, mean, std, p<PERCENTILES>."""
    header_cols = (
        ["$f_s$ (Hz)", "$\hat f_s$ (Hz)", "Error (ppm)",
         r"Mean PDV ($\mu$s)", r"Std PDV ($\mu$s)", r"Max PDV ($\mu$s)"]
    )
    col_spec = "l" + "c" * (len(header_cols) - 1)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Timing statistics per nominal sample rate}",
        r"\label{tab:timing_stats}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        ppm = (row['true_fs'] - row['nominal_fs']) / row['nominal_fs'] * 1e6
        cells = [
            f"{row['nominal_fs']:.0f}",
            f"{row['true_fs']:.3f}",
            f"{ppm:.1f}",
            f"{row['mean']:.3f}",
            f"{row['std']:.3f}",
            f"{row['max']:.3f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "pdv_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No files matched '{pattern}'")
        sys.exit(1)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    rows = []
    pdv_by_fs = []
    for path in paths:
        nominal_fs, true_fs, sample_counter, pdv_us, _inter_sample_us = load_pdv(path)
        print(f"{os.path.basename(path)}: nominal_fs={nominal_fs:.0f} Hz, "
              f"true_fs={true_fs:.4f} Hz, N={len(pdv_us)}")

        plot_pdv_over_time(nominal_fs, true_fs, sample_counter, pdv_us)
        pdv_by_fs.append((nominal_fs, pdv_us))

        row = {"nominal_fs": nominal_fs, "true_fs": true_fs}
        row.update(pdv_stats(pdv_us))
        rows.append(row)

    pdv_by_fs.sort(key=lambda t: t[0])
    plot_pdv_histogram(pdv_by_fs)

    rows.sort(key=lambda r: r["nominal_fs"])
    table_path = os.path.join(PLOTS_DIR, "timing_stats.tex")
    save_latex_table(rows, table_path)
    print(f"\nSaved {len(rows)} figure(s) to {PLOTS_DIR} and table to {table_path}")


if __name__ == "__main__":
    main()
