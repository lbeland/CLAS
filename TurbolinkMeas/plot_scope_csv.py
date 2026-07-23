"""
Plot an oscilloscope CSV export (time, voltage).

Expected CSV format (2 header rows + data rows):
    x-axis,1
    second,Volt
    -25.04420E-03,-163.01499E-03
    ...

Usage:
    python3 plot_scope_csv.py /path/to/scope.csv
"""

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = sys.path[0]  # directory of this script, used as default path for the CSV


def load_scope_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)                 # header row, e.g. ["x-axis", "1"]
        units = next(reader)         # units row, e.g. ["second", "Volt"]
        data  = np.array([[float(row[0]), float(row[1])]
                          for row in reader if len(row) == 2 and row[0] and row[1]])
    time, volt = data[:, 0], data[:, 1]
    return time, volt, units


def find_falling_edges(time, volt, high_frac=0.7, low_frac=0.3):
    """Find top-to-down (high-to-low) flank times using a hysteresis (Schmitt) threshold."""
    high_thresh = 4
    low_thresh  = 2

    edges = []
    is_high = volt[0] > high_thresh
    for i in range(1, len(volt)):
        if is_high and volt[i] < low_thresh:
            edges.append(i-1)
            is_high = False
        elif not is_high and volt[i] > high_thresh:
            is_high = True
    return time[edges]

def find_rising_edges(time, volt, high_frac=0.7, low_frac=0.3):
    """Find bottom-to-up (low-to-high) flank times using a hysteresis (Schmitt) threshold."""
    high_thresh = 4
    low_thresh  = 2

    edges = []
    is_low = volt[0] < low_thresh
    for i in range(1, len(volt)):
        if is_low and volt[i] > high_thresh:
            edges.append(i-1)
            is_low = False
        elif not is_low and volt[i] < low_thresh:
            is_low = True
    return time[edges]


def plot_scope_data(time, volt, units, path, falling_edges):
    plt.figure(figsize=(10, 4))
    plt.plot(time*1e6, volt, linewidth=0.8)
    for edge_time in falling_edges:
        plt.axvline(edge_time*1e6, color="tab:red", linewidth=0.8, linestyle="--", alpha=0.6)
    plt.xlabel(f"Time (us)")
    plt.ylabel(f"Voltage ({units[1]})")
    plt.title(os.path.basename(path))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot an oscilloscope CSV export (time, voltage)")
    parser.add_argument("path", nargs="?", default="/media/linda/FIDIS STICK/scope_2.csv",
                        help="Path to the scope CSV file (default: ./scope.csv)")
    args = parser.parse_args()

    path = args.path if os.path.exists(args.path) else os.path.join(ROOT, args.path)
    try:
        time, volt, units = load_scope_csv(path)
    except (FileNotFoundError, OSError):
        print(f"Could not find '{args.path}'. Pass the path to your CSV as the first argument:\n"
              f"  python3 plot_scope_csv.py /path/to/scope.csv")
        sys.exit(1)

    print(f"Loaded {len(time)} samples from '{path}'")

    falling_edges = find_falling_edges(time, volt)
    print(f"Found {len(falling_edges)} falling (top-to-down) edges at ({units[0]}):")
    print(falling_edges)
    diff = np.diff(falling_edges)
    print(f"  Intervals: {[f'{d*1e6:.5f}' for d in diff]}")

    rising_edges = find_rising_edges(time, volt)
    print(f"Found {len(rising_edges)} rising (bottom-to-up) edges at ({units[0]}):")
    print(rising_edges)
    diff_rising = np.diff(rising_edges)
    print(f"  Intervals: {[f'{d*1e6:.5f}' for d in diff_rising]}")

    plot_scope_data(time, volt, units, path, falling_edges)


if __name__ == "__main__":
    main()
