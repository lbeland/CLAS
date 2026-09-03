"""
ROC-style comparison of approve_peak rejection strategies. Was plot_strategy_roc.py.

Plots false_positive_rate_no_peak (x) vs false_negative_rate_with_peak (y)
for every strategy in outputs/reject/strategy_comparison_summary.csv. The
Pareto-optimal strategies (no other strategy beats them on BOTH axes
simultaneously) are highlighted and connected as the empirical frontier --
pick your operating point from that frontier based on how costly a false
positive is relative to a false negative for your actual use case, rather
than from a single blended metric.
"""
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from iaf_compare.paths import REJECT_DIR, ensure_output_dirs


def pareto_frontier(df, x_col, y_col):
    """Return the subset of rows not dominated by any other row (lower is
    better on both axes). A point is dominated if another point is <= on
    both axes and strictly < on at least one."""
    pts = df[[x_col, y_col]].values
    is_dominated = [
        any(
            (pts[j][0] <= pts[i][0] and pts[j][1] <= pts[i][1])
            and (pts[j][0] < pts[i][0] or pts[j][1] < pts[i][1])
            for j in range(len(pts)) if j != i
        )
        for i in range(len(pts))
    ]
    frontier = df.loc[[not d for d in is_dominated]].copy()
    return frontier.sort_values(x_col)


def plot_strategy_roc(csv_path, output_path,
                      x_col="false_positive_rate_no_peak",
                      y_col="false_negative_rate_with_peak",
                      label_col="strategy"):
    df = pd.read_csv(csv_path)
    frontier = pareto_frontier(df, x_col, y_col)
    on_frontier = df[label_col].isin(frontier[label_col])

    fig, ax = plt.subplots(figsize=(9, 7))

    # Dominated points: muted, small
    ax.scatter(df.loc[~on_frontier, x_col], df.loc[~on_frontier, y_col],
               s=60, color="#9aa5b1", alpha=0.7, zorder=2,
               label="dominated (a better option exists)")

    # Pareto frontier: highlighted, connected, larger
    ax.plot(frontier[x_col], frontier[y_col], color="#d64550",
            linewidth=1.5, linestyle="--", zorder=3, alpha=0.8)
    ax.scatter(frontier[x_col], frontier[y_col], s=110, color="#d64550",
               edgecolor="white", linewidth=1.2, zorder=4,
               label="Pareto frontier (no strategy beats these on both axes)")

    # Label frontier points prominently; label dominated points only with a
    # lighter touch and slight jitter-free offset alternation to reduce
    # overlap in dense clusters.
    seen_positions = defaultdict(int)

    for _, row in df.iterrows():
        is_front = row[label_col] in frontier[label_col].values
        short_label = row[label_col].replace("_plus", "").replace("_only", "")

        # stack labels vertically when multiple points share (almost) the
        # same coordinates, instead of letting them print on top of each other
        pos_key = (round(row[x_col], 3), round(row[y_col], 3))
        stack_idx = seen_positions[pos_key]
        seen_positions[pos_key] += 1
        y_jitter = stack_idx * 11

        ax.annotate(
            short_label,
            (row[x_col], row[y_col]),
            textcoords="offset points", xytext=(6, 4 + y_jitter),
            fontsize=7.5 if is_front else 6.5,
            color="#d64550" if is_front else "#7c8187",
            fontweight="bold",
            alpha=1.0,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    ax.set_xlabel("False positive rate (no-peak conditions)\n→ worse, more false alarms", fontsize=11)
    ax.set_ylabel("False negative rate (with-peak conditions)\n→ worse, more missed peaks", fontsize=11)
    ax.set_title("Peak-rejection strategy comparison", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    # Mark the origin (0,0) as the unreachable ideal, for visual reference
    ax.scatter([0], [0], marker="*", s=250, color="gold",
               edgecolor="black", linewidth=0.8, zorder=5)
    ax.annotate("ideal", (0, 0), textcoords="offset points", xytext=(8, -10),
                fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved to {output_path}")
    print("\nPareto-optimal strategies (sorted by false_positive_rate):")
    print(frontier[[label_col, x_col, y_col, "mae_with_peak"]].to_string(index=False))

    return frontier


if __name__ == "__main__":
    ensure_output_dirs()
    plot_strategy_roc(REJECT_DIR / "strategy_comparison_summary.csv",
                      REJECT_DIR / "strategy_roc.png")
