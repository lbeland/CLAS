import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import PathPatch
import seaborn as sns
from pathlib import Path
import h5py

BASE_FOLDER = Path(__file__).parent
PALETTE = sns.color_palette("tab10")

def params_excluding(base_params, exclude_key):
    """Return a copy of base_params with exclude_key removed if present.

    This lets us reuse `default_filter` while allowing the removed key
    to be used as a hue in plotting calls.
    """
    p = dict(base_params)
    p.pop(exclude_key, None)
    return p

def load_metrics(path="iaf_results.h5"):
    """Load just the aggregated metrics into a DataFrame."""
    rows = []
    with h5py.File(path, "r") as hf:
        for cond_id, grp in hf["conditions"].items():
            config = dict(grp.attrs)
            for algo, algo_grp in grp.items():
                if algo in ["ground_truth", "first_window", "spectrum_psd", "spectrum_freq_bins"]:
                    continue
                rows.append({
                    "condition_id": cond_id,
                    **config,
                    "algorithm": algo,
                    **dict(algo_grp.attrs)
                })
    return pd.DataFrame(rows)

def load_samples_for_plot(hdf_path, df_metrics, **filter_kwargs):
    """
    Filter conditions first, then load only matching samples from HDF5.
    Much more efficient than loading everything and filtering after.
    """
    df_filtered = filter_df(df_metrics, **filter_kwargs)
    
    rows = []
    with h5py.File(hdf_path, "r") as hf:
        for _, row in df_filtered.iterrows():
            cond_id = f"{int(row['condition_id']):04d}"
            algo = row["algorithm"]
            try:
                errors = hf[f"conditions/{cond_id}/{algo}/errors"][:]
                if len(errors) == 0:
                    errors = np.array([np.nan])  # To ensure we still get a row for this condition
                estimates = hf[f"conditions/{cond_id}/{algo}/estimates"][:]
                row_dict = row.to_dict()
                for err, est in zip(errors, estimates):
                    rows.append({**row_dict, "error": err,
                                 "abs_error": abs(err), "estimate": est})
            except KeyError:
                pass
    return pd.DataFrame(rows)

def load_spectra_by_hue(hdf_path, df_filtered, hue):
    """
    Load first found spectrum per unique hue value.
    Returns dict: hue_value -> (freq_bins, mean_psd)
    """
    spectra = {}
    with h5py.File(hdf_path, "r") as hf:
        for hue_val, group in df_filtered.groupby(hue):
            psds = []
            freq_bins = None
            seen = set()
            for _, row in group.iterrows():
                cond_id = f"{int(row['condition_id']):04d}"
                if cond_id in seen:
                    continue
                seen.add(cond_id)
                try:
                    grp = hf[f"conditions/{cond_id}"]
                    psds.append(grp["spectrum_psd"][:])
                    if freq_bins is None:
                        freq_bins = grp["spectrum_freq_bins"][:]
                except KeyError:
                    pass
            if psds:
                spectra[hue_val] = (freq_bins, psds[0])
    return spectra

def load_timeseries_by_hue(hdf_path, df_filtered, hue):
    """
    Load first found timeseries per unique hue value.
    Returns dict: hue_value -> (freq_bins, mean_psd)
    """
    window = {}
    with h5py.File(hdf_path, "r") as hf:
        for hue_val, group in df_filtered.groupby(hue):
            timeseries = []
            seen = set()
            for _, row in group.iterrows():
                cond_id = f"{int(row['condition_id']):04d}"
                if cond_id in seen:
                    continue
                seen.add(cond_id)
                try:
                    grp = hf[f"conditions/{cond_id}"]
                    timeseries.append(grp["first_window"][:])
                except KeyError:
                    pass
            if timeseries:
                window[hue_val] = timeseries[0]
    return window

def filter_df(df, **kwargs):
    """
    Filter DataFrame to rows matching all keyword conditions.

    Examples
    --------
    filter_df(df, noise_type="none", mod_freq=0.0, mod_amp=0.5)
    filter_df(df, noise_type=["pink", "white"])  # multiple allowed values
    """
    mask = pd.Series(True, index=df.index)
    for col, val in kwargs.items():
        if isinstance(val, (list, tuple)):
            mask &= df[col].isin(val)
        else:
            mask &= (df[col] == val)
    return df[mask]

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

    print(f"\n=== Detection Rates ===")
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
             title=None, freq_max=30):
 
    if hue is not None:
        fig = plt.figure(figsize=(15, 6))
        # Split figure into left (1/3) and right (2/3)
        gs = gridspec.GridSpec(1, 2, figure=fig,
                               width_ratios=[1, 2.5], wspace=0.1, hspace=0.1)
 
        # Split left column into top (timeseries) and bottom (spectrum)
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0],
                                                   hspace=0.4)
        ax_ts   = fig.add_subplot(gs_left[0])
        ax_spec = fig.add_subplot(gs_left[1])
        ax_box  = fig.add_subplot(gs[1])
 
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
            ax_spec.semilogy(freq_bins[mask], mean_psd[mask],
                         color=color, linewidth=1.4, alpha=0.85, label=str(hue_val))
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.set_ylabel("log(Power)")
        ax_spec.set_title("Example Spectrum")
 
    else:
        fig, ax_box = plt.subplots(figsize=(10, 5))
        palette = PALETTE
 
    # --- Box plot panel ---
    # x_order = ["stupid_max", "parabolic_max", "fooof","philistine","combine"]
    x_order = ["stupid_max", "fooof","philistine","combine_complex","combine_simple"]
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
        # fp/fn/n are stored once per (condition_id, algorithm) and broadcast onto
        # every sample row of that condition. Multiple condition_ids (noise
        # realizations) share the same (x, hue) combo, so the true fail rate for
        # a box must pool fp/fn/n across all of those condition_ids first
        # (drop duplicate condition_id rows so each contributes once), THEN
        # divide -- not average the per-condition fail_rates.
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
    ax_box.set_ylabel(y)
    ax_box.set_title(title if title else f"{y} distribution",fontdict={"fontsize": 18})
    
    # Adjust layout to make room for bottom legend (only when hue is present)
    if hue is not None:
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.25)
    else:
        fig.subplots_adjust(left=0.06, right=0.98)

    plt.savefig(f"{BASE_FOLDER}/plots/{hue}.svg", dpi=300, bbox_inches="tight")

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

if __name__ == "__main__":
    HDF_PATH = BASE_FOLDER / "iaf_results.h5"
    df_metrics = load_metrics(HDF_PATH)  # fast, always load this
    default_filter =  {
        "carrier_freq":         10.32,
        # Peak shape in frequency domain
        "carrier_waveform":     "gaussian",   # "gaussian" | "sine" | "burst"
        # Aperiodic component
        "aperiodic_exponent":   2.0,          # β — slope of 1/f^β
        # Peak(s)
        "n_peaks":              1,
        "peak_bw":              0.5,          # Gaussian σ in Hz
        # KEY PARAM: peak power relative to aperiodic floor at aperiodic_ref_freq
        "peak_snr_db":          10.0,
        # Analysis
        "window_length_sec":    10,
    }
    error_label = "error"

    # Each plot loads only what it needs
    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "window_length_sec")),
        HDF_PATH,
        x="algorithm", y=error_label, hue="window_length_sec",
        title="Effect of Window Length\n"
    )


    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "carrier_freq")),
        HDF_PATH,
        x="algorithm", y=error_label, hue="carrier_freq",
        title="Effect of Carrier Frequency\n"
    )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "peak_snr_db")),
        HDF_PATH,
        x="algorithm", y=error_label, hue="peak_snr_db",
        title="Effect of Peak SNR\n"
    )
    

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "carrier_waveform")),
        HDF_PATH,
        x="algorithm", y=error_label, hue="carrier_waveform",
        title="Effect of Carrier Waveform\n"
    )

    # plot_box(
    #     load_samples_for_plot(HDF_PATH, df_metrics,
    #                           **params_excluding(default_filter, "mod_freq")),
    #     HDF_PATH,
    #     x="algorithm", y=error_label, hue="mod_freq",
    #     title="Effect of Modulation Frequency\n"
    # )

    # plot_box(
    #     load_samples_for_plot(HDF_PATH, df_metrics,
    #                           **params_excluding({**default_filter.copy(), "mod_freq": 0.01}, "mod_amp")),
    #     HDF_PATH,
    #     x="algorithm", y=error_label, hue="mod_amp",
    #     title="Effect of Modulation Amplitude\n"
    # )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "peak_bw")),
        HDF_PATH,
        x="algorithm", y=error_label, hue="peak_bw",
        title="Effect of Peak Bandwidth\n"
    )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "n_peaks")),
        HDF_PATH,
        x="algorithm", y=error_label, hue="n_peaks",
        title="Effect of Number of Peaks\n"
    )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "aperiodic_exponent")),
        HDF_PATH,
        x="algorithm", y=error_label, hue="aperiodic_exponent",
        title="Effect of Aperiodic Exponent\n"
    )

    # # --- Example: Plot error distribution over all conditions ---
    # plot_box(df_metrics, x="algorithm", y="mae", hue="noise_type",
    #     title="MAE distribution across Noise Types")
    # plot_box(df_metrics, x="algorithm", y="mae", hue="noise_level",
    #     title="MAE distribution across Noise Levels")
    # plot_box(df_metrics, x="algorithm", y="mae", hue="mod_freq",
    #     title="MAE distribution across Modulation Frequencies")
    # plot_box(df_metrics, x="algorithm", y="mae", hue="mod_amp",
    #     title="MAE distribution across Modulation Amplitudes")
    # plot_box(df_metrics, x="algorithm", y="mae", hue="carrier_freq",
    #     title="MAE distribution across IAF Frequencies")
    
    plot_box(load_samples_for_plot(HDF_PATH, df_metrics), HDF_PATH, x="algorithm", y=error_label, title="Error distribution across all conditions"),

    # Print mean and std of MAE for each algorithm
    summary = (df_metrics.groupby("algorithm")["mae"]
               .agg(["median","mean", "std"])
               .mul(1000)  # Convert to mHz for readability
               .round(2))
    print("\n=== MAE Summary (in mHz) ===")
    print(summary.to_string())

    # # Print Conditions where " + error_label.upper() + " > 4Hz for any algorithm
    # high_error = df_metrics[df_metrics[error_label] > 4]
    # print("Conditions with " + error_label.upper() + " > 4Hz:")
    # print(high_error[["algorithm", error_label] + [col for col in df_metrics.columns if col not in ['condition_id', 'algorithm', error_label,"fs", "signal_length_sec", "freq_range", "alpha_band",
    #     "SG_window", "SG_poly", "pink_ax_r2"]]])

    # plt.show()