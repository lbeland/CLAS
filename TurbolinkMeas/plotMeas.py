"""
Analyze the empirical jitter.csv produced by measurement.cpp.

Expected CSV format (header + rows):
    sample_counter,jitter_us

This script:
  1. Loads the empirical jitter distribution (already detrended / floor-zeroed
     by measurement.cpp, i.e. min(jitter) == 0 by construction there).
  2. Reports summary statistics and percentiles.
  3. Fits/compares candidate distributions (Uniform, Exponential, truncated
     Normal) against the empirical histogram, purely descriptively -- this
     does NOT re-derive the physical floor (that was already done in
     measurement.cpp); it helps you understand the *shape* of the jitter
     so you can reason about how many calibration packets you need.
  4. Runs the convergence analysis from verify_calibration.py but using
     resampling from your REAL empirical jitter values instead of a
     synthetic distribution, answering: "given my actual measured jitter,
     how many calibration packets do I need for the start_time_us_
     estimate to be accurate to within X us?"

Usage:
    python3 analyze_jitter.py /path/to/jitter.csv
    python3 analyze_jitter.py /path/to/jitter.csv --fs 10000

If no path is given, looks for ./jitter.csv in the current directory.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress

ROOT = sys.path[0]  # directory of this script, used as default path for jitter.csv

def load_jitter(path):
    if not os.path.exists(path):
        path = os.path.join(ROOT, path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    if "jitter_us" not in data.dtype.names:
        raise ValueError(
            f"Expected a 'jitter_us' column, found columns: {data.dtype.names}"
        )
    sample_counter = data["sample_counter"]
    jitter_us = data["jitter_us"]
    inter_sample_us = data["inter_sample_us"]
    return sample_counter, jitter_us, inter_sample_us


def summary_stats(jitter_us):
    print("--- Empirical jitter summary ---")
    print(f"  N samples:        {len(jitter_us)}")
    print(f"  mean:              {jitter_us.mean():.3f} us")
    print(f"  std:               {jitter_us.std():.3f} us")
    print(f"  min:               {jitter_us.min():.3f} us")
    print(f"  max:               {jitter_us.max():.3f} us")
    for p in [5, 25, 75, 90, 95, 99]:
        print(f"  p{p:<5}:            {np.percentile(jitter_us, p):.3f} us")
    print()


def shape_diagnostics(jitter_us):
    """
    Purely descriptive comparison against a few simple one-sided shapes.
    This does not change your calibration -- it's to help you understand
    whether jitter looks closer to uniform, exponential, or something
    heavier-tailed, since that affects how many calibration packets you
    need for a given target accuracy (see convergence section below).
    """
    mean = jitter_us.mean()
    std = jitter_us.std()
    median = np.median(jitter_us)

    # Uniform(0, 2*mean) has std = mean / sqrt(3) ~ 0.577 * mean
    uniform_std_ratio = std / mean if mean > 0 else np.nan
    # Exponential(mean) has std == mean (ratio == 1.0)
    # A ratio << 1 suggests sub-exponential / closer to uniform (bounded, light tail)
    # A ratio >= 1 suggests exponential-like or heavier-tailed

    skewness = np.mean((jitter_us - mean) ** 3) / (std ** 3) if std > 0 else np.nan

    print("--- Shape diagnostics (descriptive only) ---")
    print(f"  mean/median ratio:      {mean / median:.3f}  "
          f"(~1.0 for symmetric-ish, >1 for right-skewed)")
    print(f"  std/mean ratio:         {uniform_std_ratio:.3f}  "
          f"(~0.577 matches Uniform(0,2*mean); ~1.0 matches Exponential(mean))")
    print(f"  skewness:               {skewness:.3f}  "
          f"(0 = symmetric, >0 = right-skewed/long tail, typical for network jitter)")
    print()


def convergence_via_resampling(jitter_us, calib_sizes, n_trials=2000, seed=0):
    """
    Empirically answers: "if I calibrate using N packets drawn from MY
    measured jitter distribution, how far off is min(jitter_sample) from
    the true floor (0, since jitter_us here is already floor-zeroed) on
    average / in the worst case?"

    This resamples (with replacement) from the empirical jitter values,
    which respects whatever shape your real distribution has -- no
    parametric assumption needed.
    """
    rng = np.random.default_rng(seed)
    print("--- Calibration convergence (resampled from YOUR empirical jitter) ---")
    print(f"  (Lower 'floor error' = better. This is how much start_time_us_")
    print(f"   would overestimate the true floor, using N calibration packets.)\n")

    print(f"  {'N':>8}  {'mean error (us)':>16}  {'p95 error (us)':>16}  {'max error (us)':>16}")
    for n in calib_sizes:
        errors = np.empty(n_trials)
        for t in range(n_trials):
            sample = rng.choice(jitter_us, size=n, replace=True)
            errors[t] = sample.min()  # true floor is 0 here, so error == min(sample)
        print(f"  {n:>8}  {errors.mean():>16.3f}  {np.percentile(errors, 95):>16.3f}  {errors.max():>16.3f}")
    print()


def plot_data(sample_counter, jitter_us, inter_sample_us):
    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(sample_counter-sample_counter[0], inter_sample_us, marker=".", linestyle="none", alpha=0.5)
    linear_fit = linregress(sample_counter, inter_sample_us)
    lin = linear_fit.intercept + linear_fit.slope * sample_counter
    ax1.plot(sample_counter-sample_counter[0], lin)
    plt.xlabel("Sample Counter")
    plt.ylabel("Inter-sample Interval (us)")
    plt.grid(True)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(sample_counter-sample_counter[0], jitter_us, marker=".", linestyle="none", alpha=0.5)
    linear_fit = linregress(sample_counter, jitter_us)
    lin = linear_fit.intercept + linear_fit.slope * sample_counter
    ax2.plot(sample_counter-sample_counter[0], lin)
    plt.xlabel("Sample Counter")
    plt.ylabel("Jitter (us)")
    plt.grid(True)
    plt.show()

def plot_all():

    fig, ax = plt.subplots(figsize=(15, 8))

    # Create histogram axis on the right
    divider = make_axes_locatable(ax)
    ax_hist = divider.append_axes("right", size="20%", pad=0.1)

    for freq in [500,1000, 5000, 10000]:
        sample_counter,_, recv_times = load_jitter(f"jitter_{freq}.csv")
        sample_counter = sample_counter - sample_counter[0]  # zero the sample counter
        x = np.linspace(0, len(sample_counter)/freq, len(sample_counter))  # convert to seconds
        recv_times = recv_times / 1e3  # convert to ms

        ax_hist.hist(recv_times, bins=25, histtype='step', orientation='horizontal', label=freq)
        ax.plot(x, recv_times, label=freq)

    # recv_times = recv_times - 1e3/freq_value

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Receive Period (ms)")

    ax.minorticks_on()
    ax_hist.minorticks_on()
    ax_hist.grid(which="both",axis="y")
    ax.grid(which="both",axis="both")
    ax.legend()
    plt.savefig("TurboLink.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze empirical jitter.csv")
    parser.add_argument("path", nargs="?", default="jitter_10000.csv",
                        help="Path to jitter.csv (default: ./jitter.csv)")
    parser.add_argument("--fs", type=float, default=10000.0,
                        help="Sample rate in Hz, used only for context in printed output")
    parser.add_argument("--target-error-us", type=float, default=20.0,
                        help="Target floor-estimation accuracy (us) for the recommendation")
    args = parser.parse_args()

    try:
        sample_counter, jitter_us, inter_sample_us = load_jitter(args.path)
    except (FileNotFoundError, OSError):
        print(f"Could not find '{args.path}'. Pass the path to your jitter.csv "
              f"as the first argument:\n  python3 analyze_jitter.py /path/to/jitter.csv")
        sys.exit(1)

    print(f"Loaded {len(jitter_us)} jitter samples from '{args.path}' "
          f"(fs={args.fs:.0f} Hz, interval={1e6/args.fs:.2f} us)\n")

    plot_data(sample_counter, jitter_us, inter_sample_us)
    summary_stats(jitter_us)

    plot_all()

    # shape_diagnostics(jitter_us)
    # calib_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    # calib_sizes = [n for n in calib_sizes if n <= len(jitter_us)]
    # convergence_via_resampling(jitter_us, calib_sizes)


if __name__ == "__main__":
    main()