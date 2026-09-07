"""
Score, store and plot the reject_compare.py benchmark outputs. Merges the old
plot_roc.py with the scoring/table code that used to live in reject_compare.py.

reject_compare.py runs the (slow) approve_peak strategy sweep and dumps one raw
row per (strategy, condition) to REJECT_DIR/strategy_comparison_raw.csv. This
script turns that CSV into everything downstream, so the sweep and the
scoring/formatting can be re-run independently (tweak a caption, a metric or a
plot style here and regenerate in seconds). Numeric results stay project-local
in REJECT_DIR; the LaTeX table and the plot go to the shared thesis plots
folder (TABLE_DIR / FIGURE_DIR), next to plot_sweep.py's outputs:

  * REJECT_DIR/strategy_comparison_summary.csv -- headline FP / FN / MAE
    tradeoff, one row per approve_peak strategy, sorted by no-peak
    false-positive rate.
  * TABLE_DIR/strategy_comparison_reject.tex -- LaTeX table of detection
    sensitivity, specificity and balanced accuracy over the five thesis
    algorithms (sorted by balanced accuracy), framed and labelled like the
    tables in plot_sweep.py.
  * FIGURE_DIR/strategy_roc.pdf / .pgf -- ROC-style plot of false_positive_rate_no_peak
    (x) vs false_negative_rate_with_peak (y) for every strategy. The
    Pareto-optimal strategies (no other strategy beats them on BOTH axes
    simultaneously) are highlighted and connected as the empirical frontier --
    pick your operating point from that frontier based on how costly a false
    positive is relative to a false negative for your actual use case, rather
    than from a single blended metric.

Run with: python plot_reject.py
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib as mpl

from iaf_compare.paths import REJECT_DIR, FIGURE_DIR, TABLE_DIR, ensure_output_dirs
from iaf_compare.plot_style import FIG_WIDTH

# Saving to a ".pgf" filename invokes the pgf backend automatically, so the
# default (interactive) backend stays active and plt.show() keeps working.
# pgf.texsystem defaults to xelatex, which isn't installed -- pdflatex is.
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Numeric results (project-local, next to iaf_results.h5) ...
RAW_CSV = REJECT_DIR / "strategy_comparison_raw.csv"
SUMMARY_CSV = REJECT_DIR / "strategy_comparison_summary.csv"
# ... plots + LaTeX table go to the shared thesis plots folder.
ACCURACY_F1_TEX = TABLE_DIR / "strategy_comparison_reject.tex"
ROC_PDF = FIGURE_DIR / "strategy_roc.pdf"

# Scored once by reject_compare.py's reference-set pass. These share the
# "strategy" column with the approve_peak variants but aren't part of the
# approve_peak sweep, so the headline summary (a head-to-head of approve_peak
# strategies) skips them; the detection-metrics table keeps them.
REFERENCE_SET = ("Maximum", "FOOOF", "RestingIAF", "alpha_fast_mt")

# The approve_peak strategy that reproduces iaf_compare.algorithms.approve_peak
# verbatim (BIC test on psd_flat, full freq_range, floor_value 0.0) -- the only
# alpha_fast row shown in the LaTeX detection-metrics table.
DEFAULT_BIC_STRATEGY = "bic_only_psd_flat_full"

# Display labels for the LaTeX detection-metrics table, mirroring
# plot_sweep.DISPLAY_NAMES / ALGO_KEYS so it reads the same as the thesis's
# other algorithm tables.
ALGO_DISPLAY = {
    "Maximum":       "Maximum",
    "FOOOF":         "FOOOF",
    "RestingIAF":    "RestingIAF",
    "alpha_fast":    r"$\alpha$-FAST",
    "alpha_fast_mt": r"$\alpha$-FAST-MT",
}


# ---------------------------------------------------------------------------
# Scoring / tables (moved out of reject_compare.py)
# ---------------------------------------------------------------------------
def write_headline_summary(df, path=SUMMARY_CSV):
    """Headline comparison: pool across seeds AND across all no-peak conditions
    to get one false-positive rate per strategy, and pool across all with-peak
    conditions to get one false-negative rate + mae per strategy. This is the
    core tradeoff the approve_peak sweep is optimizing. Writes ``path`` and
    returns the summary DataFrame."""
    no_peak = df[df["n_peaks"] == 0]
    with_peak = df[df["n_peaks"] > 0]

    strategy_names = [s for s in df["strategy"].unique() if s not in REFERENCE_SET]

    summary_rows = []
    for strategy_name in strategy_names:
        np_grp = no_peak[no_peak["strategy"] == strategy_name]
        wp_grp = with_peak[with_peak["strategy"] == strategy_name]

        fp_total, n_np_total = np_grp["fp"].sum(), np_grp["n"].sum()
        fn_total, n_wp_total = wp_grp["fn"].sum(), wp_grp["n"].sum()
        fp_wp_total = wp_grp["fp"].sum()  # fp can still happen in n_peaks>0
        #                                   conditions with n_peaks>1 if one of
        #                                   several peaks is spuriously found
        #                                   where none exists in that window --
        #                                   keep separate from fn
        mae_vals = wp_grp["mae"].dropna()

        summary_rows.append({
            "strategy": strategy_name,
            "false_positive_rate_no_peak": fp_total / n_np_total if n_np_total else np.nan,
            "false_negative_rate_with_peak": fn_total / n_wp_total if n_wp_total else np.nan,
            "fp_rate_within_with_peak": fp_wp_total / n_wp_total if n_wp_total else np.nan,
            "mae_with_peak": mae_vals.mean() if len(mae_vals) else np.nan,
            "n_no_peak_samples": n_np_total,
            "n_with_peak_samples": n_wp_total,
        })

    summary = pd.DataFrame(summary_rows).sort_values("false_positive_rate_no_peak")
    summary.to_csv(path, index=False)
    print("\n=== Summary (sorted by false positive rate) ===")
    print(summary.to_string(index=False))
    return summary


def write_detection_table(df, path=ACCURACY_F1_TEX, window_length_sec=10):
    """LaTeX table of peak-detection sensitivity, specificity and balanced
    accuracy for the five thesis algorithms, pooling the confusion-matrix counts
    across every condition and seed:

        sensitivity       = TP / (TP + FN)   -- over the with-peak conditions
        specificity       = TN / (TN + FP)   -- over the no-peak conditions
        balanced accuracy = (sensitivity + specificity) / 2

    Each rate is normalised within its own class, so balanced accuracy is not
    skewed by the benchmark's with-peak : no-peak condition imbalance (roughly
    19 : 7) the way raw accuracy / F1 would be. Rows sorted by balanced
    accuracy, descending.

    alpha_fast appears only under its default BIC test (DEFAULT_BIC_STRATEGY),
    relabelled -- the other approve_peak variants are dropped here; Maximum,
    FOOOF, RestingIAF and alpha_fast_mt come from the reference-set pass.
    Algorithm labels and the table framing (centered float + caption) match the
    other thesis tables in plot_sweep.py."""
    keep = {
        "Maximum":            "Maximum",
        "FOOOF":              "FOOOF",
        "RestingIAF":         "RestingIAF",
        DEFAULT_BIC_STRATEGY: "alpha_fast",
        "alpha_fast_mt":      "alpha_fast_mt",
    }
    missing = [s for s in keep if s not in set(df["strategy"])]
    if missing:
        print(f"WARNING: {missing} not in results -- LaTeX table will omit them")
    df = df[df["strategy"].isin(keep)].copy()
    df["strategy"] = df["strategy"].map(keep)

    grp = df.groupby("strategy")[["tp", "tn", "fp", "fn"]].sum()
    tp, tn, fp, fn = grp["tp"], grp["tn"], grp["fp"], grp["fn"]
    grp["sensitivity"] = tp / (tp + fn).where(tp + fn > 0, np.nan)
    grp["specificity"] = tn / (tn + fp).where(tn + fp > 0, np.nan)
    grp["balanced_accuracy"] = (grp["sensitivity"] + grp["specificity"]) / 2
    grp = grp.sort_values("balanced_accuracy", ascending=False)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Peak-detection sensitivity, specificity and balanced "
        rf"accuracy by algorithm, pooled over every rejection-benchmark "
        rf"condition and seed (window length = {window_length_sec}\,s). }}",
        r"\label{tab:reject_accuracy}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Algorithm & Specificity & Sensitivity & Balanced accuracy \\",
        r"\midrule",
    ]
    for name, r in grp.iterrows():
        lines.append(f"{ALGO_DISPLAY.get(name, name)} & "
                     f"{r['specificity']:.3f} & {r['sensitivity']:.3f} & "
                     f"{r['balanced_accuracy']:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))
    print(f"\nWrote LaTeX table to {path}")
    print(grp[["specificity", "sensitivity", "balanced_accuracy"]].to_string())


# ---------------------------------------------------------------------------
# ROC-style Pareto plot (was plot_roc.py)
# ---------------------------------------------------------------------------
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


def plot_strategy_roc(summary_df, output_path=ROC_PDF,
                      x_col="false_positive_rate_no_peak",
                      y_col="false_negative_rate_with_peak",
                      label_col="strategy"):
    df = summary_df
    frontier = pareto_frontier(df, x_col, y_col)
    on_frontier = df[label_col].isin(frontier[label_col])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH * 7 / 9))

    # Dominated points: muted, small
    ax.scatter(df.loc[~on_frontier, x_col], df.loc[~on_frontier, y_col],
               s=60, color="#9aa5b1", alpha=0.7, zorder=2)

    # Pareto frontier: highlighted, connected, larger
    ax.plot(frontier[x_col], frontier[y_col], color="#d64550",
            linewidth=1.5, linestyle="--", zorder=3, alpha=0.8)
    ax.scatter(frontier[x_col], frontier[y_col], s=110, color="#d64550",
               edgecolor="white", linewidth=1.2, zorder=4,
               label="Pareto frontier")

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

    ax.set_xlabel("False positive rate (no-peak conditions)")
    ax.set_ylabel("False negative rate (with-peak conditions)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    plt.tight_layout()
    stem = output_path.with_suffix("")
    plt.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(stem.with_suffix(".pgf"), bbox_inches="tight")
    plt.show()
    print(f"Saved to {stem}.pdf / {stem}.pgf")
    print("\nPareto-optimal strategies (sorted by false_positive_rate):")
    print(frontier[[label_col, x_col, y_col, "mae_with_peak"]].to_string(index=False))

    return frontier


def main():
    ensure_output_dirs()
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} raw rows from {RAW_CSV}")
    summary = write_headline_summary(df)
    write_detection_table(df)
    plot_strategy_roc(summary)


if __name__ == "__main__":
    main()
