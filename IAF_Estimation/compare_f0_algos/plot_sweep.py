"""Figures + summary tables from outputs/iaf_results.h5. Was plot_iaf_tests.py.

HDF5 loading now lives in iaf_compare.io_hdf5 (shared with the rest of the
package). Figures are written to iaf_compare.paths.FIGURE_DIR -- point
FIGURE_DIR at another directory below if you need them elsewhere.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib as mpl

from iaf_compare import config
from iaf_compare.paths import RESULTS_H5, FIGURE_DIR, TABLE_DIR, ensure_output_dirs
from iaf_compare.plot_style import FIG_WIDTH, FIGSIZE
from iaf_compare.io_hdf5 import (
    load_metrics, load_samples_for_plot, load_spectra_by_hue,
    load_timeseries_by_hue, filter_df, params_excluding,
)

# seaborn's boxplot still calls Axes.bxp(vert=...) internally, which newer
# matplotlib flags as pending deprecation -- one warning per plot_box() call,
# which floods the terminal across the whole sweep. Not actionable from here
# (it's seaborn's internal call, not ours), so silence just this warning.
warnings.filterwarnings(
    "ignore", message=r"vert: bool will be deprecated",
    category=PendingDeprecationWarning,
)

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

PALETTE = sns.color_palette("tab10")

# Internal algorithm keys (as stored in the HDF5 store / "algorithm" column)
# don't always make good display labels -- this maps the ones that need
# LaTeX rendering to their display form. Applied to a copy of the data right
# before it's plotted/printed, never before HDF5 lookups (which need the raw
# key).
DISPLAY_NAMES = {
    # underscore escaped: text.usetex is on (see mpl.rcParams below), and a
    # bare "_" outside math mode is invalid LaTeX and would fail to compile.
    "alpha_fast":    r"$\alpha$-FAST",
    "alpha_fast_mt": r"$\alpha$-FAST-MT",
}


def with_display_names(df, column="algorithm"):
    df = df.copy()
    df[column] = df[column].replace(DISPLAY_NAMES)
    return df


# Raw "algorithm" keys, in the fixed column order used by the thesis
# comparison table (and everywhere else all 5 algorithms are listed together).
ALGO_KEYS = ["Maximum", "FOOOF", "RestingIAF", "alpha_fast", "alpha_fast_mt"]

# Same fixed order, but with display names applied -- for reindexing tables that
# are assembled via groupby (which would otherwise sort them alphabetically).
ALGO_KEYS_DISPLAY = [DISPLAY_NAMES.get(k, k) for k in ALGO_KEYS]


def order_by_algo(df):
    """Reindex a DataFrame whose index is the (display-name) "algorithm" into
    the fixed ALGO_KEYS order. Algorithms missing from the frame are skipped;
    any not in ALGO_KEYS are kept, appended in their existing order."""
    order = [a for a in ALGO_KEYS_DISPLAY if a in df.index]
    order += [a for a in df.index if a not in order]
    return df.reindex(order)


def _fmt_value(val):
    if isinstance(val, float):
        return str(int(val)) if val.is_integer() else f"{val:g}"
    return str(val)


def _fmt_mean_std(mean_val, std_val, bold=False):
    if pd.isna(mean_val):
        return "--"
    body = rf"{mean_val:.2f} \pm {std_val:.2f}"
    # \mathbf bolds the digits; \pm stays rendered (math symbols are unaffected
    # by \mathbf but still print), so no extra package is needed in the thesis
    # preamble.
    return rf"$\mathbf{{{body}}}$" if bold else rf"${body}$"


def save_ae_summary_latex(summary_df, out_path, window_length_sec):
    """summary_df: index "algorithm", columns median/mean/std (Hz) of the
    per-seed absolute error pooled over every condition at this window length
    -- as built in main()'s per-window-length loop."""
    table = summary_df.reset_index().rename(columns={"algorithm": "Algorithm"})
    latex = table.to_latex(
        index=False, escape=False, float_format="%.2f",
        caption=f"Absolute error summary [Hz], window length = {window_length_sec}\\,s",
        label=f"tab:ae_summary_wl{window_length_sec}",
    )
    # to_latex omits \centering, so the tabular floats flush-left -- add it.
    latex = latex.replace(r"\begin{table}", "\\begin{table}\n\\centering", 1)
    out_path.write_text(latex)


def save_big_comparison_table(df_metrics, wl_filter, sweeps, param_names, out_path, window_length_sec):
    """One row block per swept parameter (as in ``sweeps``), one row per
    tested value, columns = all 5 algorithms. Each cell is that single
    condition's MAE +/- the std of its per-seed absolute errors [Hz], both
    pulled straight from df_metrics's pre-pooled "mae"/"std" columns (computed
    over every seed of that one condition -- NOT a mean over several MAEs), so
    no need to re-load per-seed samples. Every other param stays pinned at
    wl_filter's value (== default_filter, matching the "Effect of X" plots).

    The algorithm with the lowest MAE in each row is set in bold (ties broken
    by ALGO_KEYS order; an all-"--" row has nothing to bold).
    """
    algo_headers = [DISPLAY_NAMES.get(k, k) for k in ALGO_KEYS]
    col_spec = "ll" + "c" * len(ALGO_KEYS)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Mean $\pm$ std of the absolute error [Hz], by algorithm and"
        rf" swept-parameter value (window length = {window_length_sec}\,s)}}",
        rf"\label{{tab:big_comparison_wl{window_length_sec}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Parameter & Value & " + " & ".join(algo_headers) + r" \\",
    ]

    for param, values in sweeps.items():
        lines.append(r"\midrule")
        param_label = param_names.get(param, param)
        base_filter = params_excluding(wl_filter, param)
        for i, val in enumerate(values):
            stats = []
            for algo in ALGO_KEYS:
                sub = filter_df(df_metrics, **{**base_filter, param: val, "algorithm": algo})
                if sub.empty:
                    stats.append((np.nan, np.nan))
                else:
                    stats.append((sub["mae"].iloc[0], sub["std"].iloc[0]))

            maes = np.array([m for m, _ in stats], dtype=float)
            best_idx = int(np.nanargmin(maes)) if not np.all(np.isnan(maes)) else -1

            row_cells = [_fmt_mean_std(m, s, bold=(k == best_idx))
                         for k, (m, s) in enumerate(stats)]
            row_label = param_label if i == 0 else ""
            lines.append(f"{row_label} & {_fmt_value(val)} & " + " & ".join(row_cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")


def print_detection_table(df_metrics, **filter_kwargs):
    df = filter_df(df_metrics, **filter_kwargs)

    if df.empty:
        print(f"No conditions match filter: {filter_kwargs}")
        return

    df = with_display_names(df)
    rate_cols = ["fail_rate", "false_pos_rate", "true_neg_rate"]
    grouped = (df.groupby("algorithm")[rate_cols + ["mae"]]
                 .mean()
                 .mul({"fail_rate": 100, "false_pos_rate": 100,
                       "true_neg_rate": 100, "mae": 1})
                 .round(3))

    grouped.columns = ["Miss/FN (%)", "False alarm/FP (%)", "Correct rejection (%)", "MAE"]
    grouped = grouped[["MAE", "Miss/FN (%)", "False alarm/FP (%)", "Correct rejection (%)"]]
    grouped = order_by_algo(grouped)

    print("\n=== Detection Rates ===")
    print(f"Filter: {filter_kwargs}  |  Conditions matched: {df['condition_id'].nunique()}\n")
    print(grouped.to_string())


def make_pivot(df, index, columns, values="mae", agg="mean"):
    return df.pivot_table(index=index, columns=columns, values=values, aggfunc=agg)


def plot_line(df, x, y="mae", hue="algorithm",
              style=None, estimator="mean", ci="sd", title=None):
    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=df, x=x, y=y, hue=hue, style=style,
                 estimator=estimator, errorbar=ci)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title if title else f"{y} vs {x}")
    plt.tight_layout()


def plot_box(df, hdf_path, x="algorithm", y="mae", hue=None,
             title=None, freq_max=30, save_name=None):

    if hue is not None:
        fig = plt.figure(figsize=FIGSIZE)
        # Split figure into left (1/5) and right (4/5)
        gs = gridspec.GridSpec(1, 2, figure=fig,
                               width_ratios=[1, 4], wspace=0.1, hspace=0.1)

        # Split left column into top (timeseries) and bottom (spectrum)
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0],
                                                   hspace=0.4)
        ax_ts = fig.add_subplot(gs_left[0])
        ax_spec = fig.add_subplot(gs_left[1])
        ax_box = fig.add_subplot(gs[1])

        palette = sns.color_palette(n_colors=df[hue].nunique(), palette=PALETTE)

        # --- Timeseries panel ---
        # Each window is a *centered* crop of one shared parent signal
        # (iaf_compare.pipeline.run_window_analysis), so the shorter windows are
        # exactly nested in the longer ones about their common midpoint. Plot
        # every trace on a center-referenced time axis (t = 0 at the midpoint)
        # so those nested windows overlay instead of being left-shifted apart.
        timeseries = load_timeseries_by_hue(hdf_path, df, hue)
        for (hue_val, ts), color in zip(timeseries.items(), palette):
            fs = df[df[hue] == hue_val]["fs"].iloc[0]
            t = (np.arange(len(ts)) - len(ts) / 2) / fs
            ax_ts.plot(t, ts, color=color, linewidth=0.8,
                       label=str(hue_val), alpha=0.85)
        ax_ts.set_xlabel("Time [s], centered")
        ax_ts.set_ylabel("Amplitude")
        ax_ts.set_title("(a)")

        # --- Spectrum panel ---
        spectra = load_spectra_by_hue(hdf_path, df, hue)
        for (hue_val, (freq_bins, mean_psd)), color in zip(spectra.items(), palette):
            mask = freq_bins <= freq_max
            ax_spec.semilogy(freq_bins[mask][1:], mean_psd[mask][1:],
                             color=color, linewidth=1.4, alpha=0.85, label=str(hue_val))
        ax_spec.set_xlabel("Frequency [Hz]")
        ax_spec.set_ylabel(r"$\log_{10}$(Power)")
        ax_spec.set_title("(b)")

    else:
        fig, ax_box = plt.subplots(figsize=FIGSIZE)
        palette = PALETTE

    # --- Box plot panel ---
    if "algorithm" in df.columns:
        df = with_display_names(df)
    # alpha_fast_mt excluded from the plots (still included in the MAE
    # summary table / big comparison table, which don't go through x_order).
    x_order = ["Maximum", "FOOOF", "RestingIAF", DISPLAY_NAMES["alpha_fast"]]

    hue_order = sorted(df[hue].unique()) if hue is not None else None
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax_box, gap=0.1,
                order=x_order, hue_order=hue_order,
                palette=palette, showmeans=True,
                boxprops={"alpha": 0.9},
                medianprops={"color": "black", "linewidth": 1.3},
                meanprops={"marker": "+", "markeredgecolor": "black",
                           "markersize": "7"})

    ax_box.minorticks_on()
    ax_box.grid(axis="y", alpha=0.9)

    if ax_box.get_legend() is not None:
        handles, labels = ax_box.get_legend_handles_labels()
        ax_box.legend_.remove()  # Remove legend from box plot (we'll add a common one later if hue is present)

    # Expand ylim to make room for labels BEFORE computing offset
    y_min, y_max = ax_box.get_ylim()
    y_range = y_max - y_min
    ax_box.set_ylim(y_min, y_max + y_range * 0.15)
    y_offset = y_max + y_range * 0.02

    # x_order categories always map to integer tick positions 0, 1, 2, ...
    # regardless of which boxes actually got drawn.
    xtick_pos = {x_val: i for i, x_val in enumerate(x_order)}

    if hue is not None:
        # fp/fn/n are stored once per (condition_id, algorithm), already
        # pooled over every seed of that condition, and broadcast onto
        # every exploded sample row. Multiple condition_ids can share the
        # same (x, hue) combo, so the true fail rate must dedupe back to
        # one row per condition_id first, THEN sum fp/fn/n and divide --
        # not average the per-condition fail_rates.
        per_condition = df.drop_duplicates(subset=[x, hue, "condition_id"])
        pooled = per_condition.groupby([x, hue])[["fp", "fn", "n"]].sum()
        grouped = (pooled["fp"] + pooled["fn"]) / pooled["n"] * 100

        # seaborn skips drawing a box entirely for any (x, hue) combo with zero
        # non-NaN y-values (e.g. a true fail_rate of 0%, all true negatives).
        # That means len(patches) can be less than len(x_order)*len(hue_order),
        # and the missing box's position can't be recovered just by counting
        # patches in order. Instead, ax.containers gives one BoxPlotContainer
        # per hue level (in hue_order); each box within it sits at a constant
        # x-offset from its category's integer tick. We read that offset off
        # whichever boxes DO exist for each hue level, then apply it to every
        # x_val -- including ones with a missing box -- so every label lands
        # at the position its box *would* occupy.
        hue_offset = {}
        for hue_val, container in zip(hue_order, ax_box.containers):
            offsets = []
            for box in getattr(container, "boxes", []):
                xc = np.mean(box.get_path().vertices[:, 0])
                offsets.append(xc - round(xc))
            if offsets:
                hue_offset[hue_val] = np.mean(offsets)

        for x_val in x_order:
            for hue_val in hue_order:
                if hue_val not in hue_offset:
                    continue  # this hue level has no boxes anywhere; skip

                try:
                    fail_rate = grouped.loc[(x_val, hue_val)]
                except KeyError:
                    fail_rate = np.nan

                # if pd.notna(fail_rate):
                #     label_x = xtick_pos[x_val] + hue_offset[hue_val]
                #     ax_box.text(
                #         label_x, y_offset,
                #         f"{int(fail_rate)}"+r"\%",
                #         ha="center", va="bottom",
                #         fontsize=7, fontweight="bold",
                #         color="red" if fail_rate == 100 else "black"
                #     )

        fig.legend(handles, labels, loc="lower center", ncol=len(hue_order),
                   bbox_to_anchor=(0.5, 0.0), frameon=True, title=hue.replace("_", " "))

    else:
        per_condition = df.drop_duplicates(subset=[x, "condition_id"])
        pooled = per_condition.groupby(x)[["fp", "fn", "n"]].sum()
        grouped = (pooled["fp"] + pooled["fn"]) / pooled["n"] * 100
        # With no hue, each x category sits at its own integer tick position
        # (0, 1, 2, ...) -- no need to index into ax_box.patches, which may be
        # missing an entry for any category with zero non-NaN y-values.
        for x_val in x_order:
            try:
                fail_rate = grouped.loc[x_val]
            except KeyError:
                fail_rate = np.nan

            # if pd.notna(fail_rate):
            #     ax_box.text(
            #         xtick_pos[x_val], y_offset,
            #         f"{int(fail_rate)}"+r"\%",
            #         ha="center", va="bottom",
            #         fontsize=7, fontweight="bold"
            #     )

    ax_box.set_xlabel("")
    ax_box.set_ylabel(y + " [Hz]")
    ax_box.set_title("(c)") #title if title else f"{y} distribution", fontdict={"fontsize": 18})

    # Adjust layout to make room for bottom legend (only when hue is present)
    if hue is not None:
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.15)
    else:
        fig.subplots_adjust(left=0.06, right=0.98)

    stem = save_name or hue
    # plt.savefig(FIGURE_DIR / f"{stem}.pgf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURE_DIR / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_condition_spectra(df_metrics, hdf_path, base_filter, sweeps, param_names,
                           save_name="condition_spectra", freq_max=30,
                           alpha_band=None):
    """3x2 grid of input spectra (no estimator results): one panel per swept
    parameter, each overlaying the stored example PSD for every tested value of
    that parameter while every other parameter stays pinned at ``base_filter``
    -- the same condition slicing as the "Effect of X" box plots, but showing
    what the algorithms actually see.

    The traces are the single-seed example Welch PSD saved per condition in the
    HDF5 store (iaf_compare.io_hdf5.write_results / pipeline.run_window_analysis),
    not a seed-average.
    """
    params = list(sweeps)
    nrows, ncols = 3, 2
    assert len(params) <= nrows * ncols, \
        f"{len(params)} swept params won't fit a {nrows}x{ncols} grid"

    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, FIG_WIDTH * 1.15),
                             sharex=True)
    axes = axes.ravel()

    for ax, param in zip(axes, params):
        df_panel = filter_df(df_metrics, **params_excluding(base_filter, param))
        # dict: param value -> (freq_bins, psd), one example condition per value
        spectra = load_spectra_by_hue(hdf_path, df_panel, param)
        palette = sns.color_palette(n_colors=max(len(spectra), 1), palette=PALETTE)

        # if alpha_band is not None:
        #     ax.axvspan(*alpha_band, color="0.85", alpha=0.5, zorder=0, lw=0)

        for (hue_val, (freq_bins, psd)), color in zip(sorted(spectra.items()), palette):
            mask = freq_bins <= freq_max
            ax.semilogy(freq_bins[mask][1:], psd[mask][1:], color=color,
                        linewidth=1.2, alpha=0.6, label=_fmt_value(hue_val))

        ax.set_title(param_names.get(param, param), fontsize=9)
        ax.grid(alpha=0.3)
        if spectra:
            ax.legend(fontsize=8, loc="upper right", frameon=True,
                    labelspacing=0.3, framealpha=0.8)

    for ax in axes[len(params):]:      # blank any unused cell (none at 6 params)
        ax.set_visible(False)
    for ax in axes[len(params) - ncols:len(params)]:
        ax.set_xlabel("Frequency [Hz]")
    for ax in axes[::ncols]:
        ax.set_ylabel(r"$\log_{10}$(Power)")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{save_name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{save_name}.pgf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bar(df, x="algorithm", y="mae", agg="mean", hue=None, title=None):
    plt.figure(figsize=FIGSIZE)
    estimator = np.mean if agg == "mean" else np.median
    sns.barplot(data=df, x=x, y=y, hue=hue, estimator=estimator)
    plt.xlabel("")
    plt.ylabel(f"{agg} {y}")
    plt.title(title if title else f"{agg} {y} per {x}")
    plt.xticks(rotation=30)
    plt.tight_layout()


def main():
    ensure_output_dirs()
    HDF_PATH = RESULTS_H5
    df_metrics = load_metrics(HDF_PATH)  # fast, always load this
    default_filter = {
        "peak_freq":            10.32,
        # Peak shape in frequency domain
        "stationarity":         "constant",   # "constant" | "bursty"
        # Aperiodic component
        "aperiodic_exponent":   2.0,          # β — slope of 1/f^β
        # Peak(s)
        "n_peaks":              1,
        "peak_width":           0.5,          # Gaussian σ in Hz
        # KEY PARAM: peak power relative to aperiodic floor at aperiodic_ref_freq
        "peak_snr_db":          10.0,
        # Analysis
        "window_length_sec":    10,
        "noise_lv":             0.5,          # Std dev of the per-bin log10-power noise (gen_noise)
    }
    error_label = "error"

    # trial_source distinguishes the OFAT base sweep (always window_length_sec
    # == default) from the window-length comparison group (nested sub-windows
    # of one shared parent signal per condition/seed). Restricting to the
    # latter means wl5/wl10/wl20 are all strictly matched to the same
    # underlying signal instances -- without it, window_length_sec == default
    # (10s) would silently mix in unrelated "base" trials.
    has_trial_source = "trial_source" in df_metrics.columns
    window_length_sweep_filter = params_excluding(default_filter, "window_length_sec")
    if has_trial_source:
        window_length_sweep_filter["trial_source"] = "window_length_sweep"

    # Window length is its own cross-cutting sweep (evaluated against every
    # condition below, sharing a parent signal per condition/seed -- see
    # iaf_compare.pipeline.process_window_length_condition), not a value fixed
    # by default_filter. This one plot compares directly across window lengths;
    # everything else below is repeated separately PER window length.
    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics, **window_length_sweep_filter),
        HDF_PATH,
        x="algorithm", y=error_label, hue="window_length_sec",
        title="Effect of Window Length\n"
    )

    window_lengths_sec = sorted(df_metrics["window_length_sec"].unique())

    for window_length_sec in window_lengths_sec:
        wl_filter = {**default_filter, "window_length_sec": window_length_sec}
        if has_trial_source:
            wl_filter["trial_source"] = "window_length_sweep"
        suffix = f" (window={window_length_sec}s)"
        prefix = f"wl{window_length_sec}"

        plot_box(
            load_samples_for_plot(HDF_PATH, df_metrics,
                                  **params_excluding(wl_filter, "peak_freq")),
            HDF_PATH,
            x="algorithm", y=error_label, hue="peak_freq",
            title=f"Effect of Peak Frequency{suffix}\n",
            save_name=f"{prefix}_peak_freq",
        )

        plot_box(
            load_samples_for_plot(HDF_PATH, df_metrics,
                                  **params_excluding(wl_filter, "peak_snr_db")),
            HDF_PATH,
            x="algorithm", y=error_label, hue="peak_snr_db",
            title=f"Effect of Peak SNR{suffix}\n",
            save_name=f"{prefix}_peak_snr_db",
        )

        plot_box(
            load_samples_for_plot(HDF_PATH, df_metrics,
                                  **params_excluding(wl_filter, "noise_lv")),
            HDF_PATH,
            x="algorithm", y=error_label, hue="noise_lv",
            title=f"Effect of Noise Level{suffix}\n",
            save_name=f"{prefix}_noise_lv",
        )

        plot_box(
            load_samples_for_plot(HDF_PATH, df_metrics,
                                  **params_excluding(wl_filter, "stationarity")),
            HDF_PATH,
            x="algorithm", y=error_label, hue="stationarity",
            title=f"Effect of Stationarity{suffix}\n",
            save_name=f"{prefix}_stationarity",
        )

        plot_box(
            load_samples_for_plot(HDF_PATH, df_metrics,
                                  **params_excluding(wl_filter, "peak_width")),
            HDF_PATH,
            x="algorithm", y=error_label, hue="peak_width",
            title=f"Effect of Peak Bandwidth{suffix}\n",
            save_name=f"{prefix}_peak_width",
        )

        plot_box(
            load_samples_for_plot(HDF_PATH, df_metrics,
                                  **params_excluding(wl_filter, "aperiodic_exponent")),
            HDF_PATH,
            x="algorithm", y=error_label, hue="aperiodic_exponent",
            title=f"Effect of Aperiodic Exponent{suffix}\n",
            save_name=f"{prefix}_aperiodic_exponent",
        )

        # All (non-window-length) conditions pooled, at this window length.
        pooled_filter = {"window_length_sec": window_length_sec}
        if has_trial_source:
            pooled_filter["trial_source"] = "window_length_sweep"

        pooled_samples = load_samples_for_plot(HDF_PATH, df_metrics, **pooled_filter)

        plot_box(
            pooled_samples,
            HDF_PATH,
            x="algorithm", y=error_label,
            title=f"Error distribution across all conditions{suffix}",
            save_name=f"{prefix}_all_conditions",
        )

        # mean/median/std of the per-seed absolute error [Hz], pooled over
        # every seed of every (non-window-length) condition at this window
        # length -- one fresh pool, NOT a mean over the per-condition MAEs
        # (which weights every condition equally regardless of seed count and
        # discards the within-condition spread). NaN abs_error rows (FN/TN
        # seeds) drop out, exactly as in compute_pooled_metrics; "std" is the
        # population std (ddof=0) of the pooled absolute errors, to match it.
        summary = order_by_algo(
            with_display_names(pooled_samples)
            .groupby("algorithm")["abs_error"]
            .agg(median="median", mean="mean", std=lambda s: s.std(ddof=0))
            .round(4))
        print(f"\n=== Absolute Error Summary (in Hz), window_length_sec={window_length_sec} ===")
        print(summary.to_string())
        save_ae_summary_latex(summary, TABLE_DIR / f"ae_summary_wl{window_length_sec}.tex",
                               window_length_sec)

        save_big_comparison_table(
            df_metrics, wl_filter, config.SWEEPS, config.name_dict,
            TABLE_DIR / f"big_comparison_wl{window_length_sec}.tex",
            window_length_sec,
        )
        if window_length_sec == 10:
            # Input spectra (not results) for every swept parameter, one 3x2 figure.
            plot_condition_spectra(
                df_metrics, HDF_PATH, wl_filter, config.SWEEPS, config.name_dict,
                save_name=f"{prefix}_condition_spectra",
                alpha_band=tuple(config.FIXED["alpha_band"]),
            )


if __name__ == "__main__":
    main()
