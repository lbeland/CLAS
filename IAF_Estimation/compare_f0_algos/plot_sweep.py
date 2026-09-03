"""Figures + summary tables from outputs/iaf_results.h5. Was plot_iaf_tests.py.

HDF5 loading now lives in iaf_compare.io_hdf5 (shared with the rest of the
package). Figures are written to iaf_compare.paths.FIGURE_DIR -- point
FIGURE_DIR at another directory below if you need them elsewhere.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib as mpl

from iaf_compare.paths import RESULTS_H5, FIGURE_DIR, ensure_output_dirs
from iaf_compare.io_hdf5 import (
    load_metrics, load_samples_for_plot, load_spectra_by_hue,
    load_timeseries_by_hue, filter_df, params_excluding,
)

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

PALETTE = sns.color_palette("tab10")


def print_detection_table(df_metrics, **filter_kwargs):
    df = filter_df(df_metrics, **filter_kwargs)

    if df.empty:
        print(f"No conditions match filter: {filter_kwargs}")
        return

    rate_cols = ["fail_rate", "false_pos_rate", "true_neg_rate"]
    grouped = (df.groupby("algorithm")[rate_cols + ["mae"]]
                 .mean()
                 .mul({"fail_rate": 100, "false_pos_rate": 100,
                       "true_neg_rate": 100, "mae": 1})
                 .round(3))

    grouped.columns = ["Miss/FN (%)", "False alarm/FP (%)", "Correct rejection (%)", "MAE"]
    grouped = grouped[["MAE", "Miss/FN (%)", "False alarm/FP (%)", "Correct rejection (%)"]]

    print("\n=== Detection Rates ===")
    print(f"Filter: {filter_kwargs}  |  Conditions matched: {df['condition_id'].nunique()}\n")
    print(grouped.to_string())


def make_pivot(df, index, columns, values="mae", agg="mean"):
    return df.pivot_table(index=index, columns=columns, values=values, aggfunc=agg)


def plot_line(df, x, y="mae", hue="algorithm",
              style=None, estimator="mean", ci="sd", title=None):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x=x, y=y, hue=hue, style=style,
                 estimator=estimator, errorbar=ci)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title if title else f"{y} vs {x}")
    plt.tight_layout()


def plot_box(df, hdf_path, x="algorithm", y="mae", hue=None,
             title=None, freq_max=30, save_name=None):

    if hue is not None:
        fig = plt.figure(figsize=(15, 6))
        # Split figure into left (1/3) and right (2/3)
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
        timeseries = load_timeseries_by_hue(hdf_path, df, hue)
        for (hue_val, ts), color in zip(timeseries.items(), palette):
            fs = df[df[hue] == hue_val]["fs"].iloc[0]
            t = np.arange(len(ts)) / fs
            ax_ts.plot(t, ts, color=color, linewidth=0.8,
                       label=str(hue_val), alpha=0.85)
        ax_ts.set_xlabel("Time (s)")
        ax_ts.set_ylabel("Amplitude")
        ax_ts.set_title("Example Window")

        # --- Spectrum panel ---
        spectra = load_spectra_by_hue(hdf_path, df, hue)
        for (hue_val, (freq_bins, mean_psd)), color in zip(spectra.items(), palette):
            mask = freq_bins <= freq_max
            ax_spec.semilogy(freq_bins[mask][1:], mean_psd[mask][1:],
                             color=color, linewidth=1.4, alpha=0.85, label=str(hue_val))
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.set_ylabel("log(Power)")
        ax_spec.set_title("Example Spectrum")

    else:
        fig, ax_box = plt.subplots(figsize=(10, 5))
        palette = PALETTE

    # --- Box plot panel ---
    # x_order = ["stupid_max", "parabolic_max", "fooof","philistine","combine"]
    x_order = ["Maximum", "FOOOF", "RestingIAF", "combine_simple", "simple_mt"]
    # x_order = ["stupid_max", "fooof","philistine","combine_simple", "simple_mt"]

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

                if pd.notna(fail_rate):
                    label_x = xtick_pos[x_val] + hue_offset[hue_val]
                    ax_box.text(
                        label_x, y_offset,
                        f"{int(fail_rate)}%",
                        ha="center", va="bottom",
                        fontsize=7, fontweight="bold",
                        color="red" if fail_rate == 100 else "black"
                    )

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

            if pd.notna(fail_rate):
                ax_box.text(
                    xtick_pos[x_val], y_offset,
                    f"{int(fail_rate)}%",
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold"
                )

    ax_box.set_xlabel(x)
    ax_box.set_ylabel(y + "[Hz]")
    ax_box.set_title(title if title else f"{y} distribution", fontdict={"fontsize": 18})

    # Adjust layout to make room for bottom legend (only when hue is present)
    if hue is not None:
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.25)
    else:
        fig.subplots_adjust(left=0.06, right=0.98)

    stem = save_name or hue
    plt.savefig(FIGURE_DIR / f"{stem}.pgf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURE_DIR / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bar(df, x="algorithm", y="mae", agg="mean", hue=None, title=None):
    plt.figure(figsize=(8, 5))
    estimator = np.mean if agg == "mean" else np.median
    sns.barplot(data=df, x=x, y=y, hue=hue, estimator=estimator)
    plt.xlabel(x)
    plt.ylabel(f"{agg} {y}")
    plt.title(title if title else f"{agg} {y} per {x}")
    plt.xticks(rotation=30)
    plt.tight_layout()


def plot_facet_line(df, x, y="mae", hue="algorithm",
                    col=None, row=None, estimator="mean"):
    g = sns.FacetGrid(df, col=col, row=row, height=4, aspect=1.2, sharey=True)
    g.map_dataframe(sns.lineplot, x=x, y=y, hue=hue, estimator=estimator)
    g.add_legend()
    g.set_axis_labels(x, y)
    plt.tight_layout()


def main():
    ensure_output_dirs()
    HDF_PATH = RESULTS_H5
    df_metrics = load_metrics(HDF_PATH)  # fast, always load this
    default_filter = {
        "peak_freq":            10.32,
        # Peak shape in frequency domain
        "stationarity":         "constant",   # "constant" | "burst"
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
                                  **params_excluding(wl_filter, "n_peaks")),
            HDF_PATH,
            x="algorithm", y=error_label, hue="n_peaks",
            title=f"Effect of Number of Peaks{suffix}\n",
            save_name=f"{prefix}_n_peaks",
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

        plot_box(
            load_samples_for_plot(HDF_PATH, df_metrics, **pooled_filter),
            HDF_PATH,
            x="algorithm", y=error_label,
            title=f"Error distribution across all conditions{suffix}",
            save_name=f"{prefix}_all_conditions",
        )

        summary = (filter_df(df_metrics, **pooled_filter)
                   .groupby("algorithm")["mae"]
                   .agg(["median", "mean", "std"])
                   .mul(1000)  # Convert to mHz for readability
                   .round(2))
        print(f"\n=== MAE Summary (in mHz), window_length_sec={window_length_sec} ===")
        print(summary.to_string())


if __name__ == "__main__":
    main()
