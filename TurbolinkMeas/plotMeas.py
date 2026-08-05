"""
Analyze jitter_<fs>.csv files produced by measurement.cpp.

Each jitter_<fs>.csv (fs = nominal sample rate, embedded in the filename)
holds only the last JITTER_WINDOW_S seconds of the run (see measurement.cpp),
with a header comment line giving the sample rate estimated from the *whole*
run:
    # true_fs_hz=<value>
    sample_counter,jitter_us,inter_sample_us

This script owns all the statistics (measurement.cpp just dumps the raw
buffered jitter): for every jitter_*.csv found, it
  1. plots jitter over time in its own figure, saved as .pdf and .pgf under
     /home/linda/Documents/MA/plots, and
  2. writes one combined LaTeX table (jitter_stats.tex) with one row per
     nominal sample rate: estimated true fs, jitter mean/std/percentiles.

Usage:
    python3 plotMeas.py [glob]

`glob` defaults to "jitter_*.csv" in this script's directory.
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

PERCENTILES = (50, 95, 99)

TRUE_FS_RE = re.compile(r"true_fs_hz=([\d.eE+-]+)")
FILENAME_RE = re.compile(r"jitter_(\d+)\.csv$")


def load_jitter(path):
    """Returns (nominal_fs, true_fs, sample_counter, jitter_us, inter_sample_us)."""
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
            data["sample_counter"], data["jitter_us"], data["inter_sample_us"])


def plot_jitter_over_time(nominal_fs, true_fs, sample_counter, jitter_us):
    width = TEXTWIDTH
    height = width * ASPECT_RATIO
    fig, ax = plt.subplots(figsize=(width, height))

    t = (sample_counter - sample_counter[0]) / true_fs
    ax.plot(t, jitter_us, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Jitter ($\mu$s)")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()

    stem = f"jitter_{int(nominal_fs)}"
    fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.pgf"))
    fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.pdf"))
    plt.close(fig)


def jitter_stats(jitter_us):
    stats = {"mean": float(jitter_us.mean()), "std": float(jitter_us.std())}
    for p in PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(jitter_us, p))
    return stats


def save_latex_table(rows, out_path):
    """rows: list of dicts with nominal_fs, true_fs, mean, std, p<PERCENTILES>."""
    header_cols = (
        ["Nominal $f_s$ (Hz)", "Estimated $f_s$ (Hz)",
         r"Mean ($\mu$s)", r"Std ($\mu$s)"]
        + [rf"p{p} ($\mu$s)" for p in PERCENTILES]
    )
    col_spec = "l" + "c" * (len(header_cols) - 1)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Jitter statistics per nominal sample rate}",
        r"\label{tab:jitter_stats}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            f"{row['nominal_fs']:.0f}",
            f"{row['true_fs']:.3f}",
            f"{row['mean']:.3f}",
            f"{row['std']:.3f}",
        ] + [f"{row[f'p{p}']:.3f}" for p in PERCENTILES]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "jitter_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No files matched '{pattern}'")
        sys.exit(1)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    rows = []
    for path in paths:
        nominal_fs, true_fs, sample_counter, jitter_us, _inter_sample_us = load_jitter(path)
        print(f"{os.path.basename(path)}: nominal_fs={nominal_fs:.0f} Hz, "
              f"true_fs={true_fs:.4f} Hz, N={len(jitter_us)}")

        plot_jitter_over_time(nominal_fs, true_fs, sample_counter, jitter_us)

        row = {"nominal_fs": nominal_fs, "true_fs": true_fs}
        row.update(jitter_stats(jitter_us))
        rows.append(row)

    rows.sort(key=lambda r: r["nominal_fs"])
    table_path = os.path.join(PLOTS_DIR, "jitter_stats.tex")
    save_latex_table(rows, table_path)
    print(f"\nSaved {len(rows)} figure(s) to {PLOTS_DIR} and table to {table_path}")


if __name__ == "__main__":
    main()
