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
                               width_ratios=[1, 2.5], wspace=0.3)

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
    x_order = ["stupid_max", "parabolic_max", "fooof","philistine","combine"]
    hue_order = sorted(df[hue].unique()) if hue is not None else None
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax_box, gap=0.1,
                order=x_order, hue_order=hue_order,
                palette=palette, showmeans=True,
                boxprops={"alpha": 0.9},
                medianprops={"color": "black", "linewidth": 1.3},
                meanprops={"marker": "+", "markeredgecolor": "black",
                           "markersize": "7"})
    
    if ax_box.get_legend() is not None:
        handles, labels = ax_box.get_legend_handles_labels()
        ax_box.legend_.remove()  # Remove legend from box plot (we'll add a common one later if hue is present)
    
    # Expand ylim to make room for labels BEFORE computing offset
    y_min, y_max = ax_box.get_ylim()
    y_range = y_max - y_min
    ax_box.set_ylim(y_min, y_max + y_range * 0.15)
    y_offset = y_max + y_range * 0.02

    patches = [p for p in ax_box.patches if isinstance(p, PathPatch)]
    x_pos = [np.mean(patch.get_path().vertices[:, 0]) for patch in patches]
    x_pos.sort()

    if hue is not None:
        grouped = df.groupby([x, hue])["fail_rate"].mean()*100

        i = 0
        for x_val in x_order:
            for hue_val in hue_order:
                if i >= len(patches):
                    continue                

                fail_rate = grouped.loc[(x_val, hue_val)]
                if pd.notna(fail_rate):
                    ax_box.text(
                        x_pos[i], y_offset,
                        f"{int(fail_rate)}%",
                        ha="center", va="bottom",
                        fontsize=7, fontweight="bold",
                        color="red" if fail_rate == 100 else "black"
                    )
                i += 1
        
        fig.legend(handles, labels, loc="lower center", ncol=len(hue_order),
                   bbox_to_anchor=(0.5, 0.0), frameon=True, title=hue.replace("_", " "))

    else:
        grouped = df.groupby(x)["fail_rate"].mean()*100
        # Seaborn lays patches out in order: all hues for x[0], all hues for x[1], ...
        # so patch order matches (x_labels × hue_values) in seaborn's own ordering
        for i, x_val in enumerate(x_order):
            if i >= len(patches):
                continue

            fail_rate = grouped.loc[x_val]
            if pd.notna(fail_rate):
                x_pos = np.mean(patches[i].get_path().vertices[:, 0])

                ax_box.text(
                    x_pos, y_offset,
                    f"{int(fail_rate)}%",
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold"
        )
                
    ax_box.set_xlabel(x)
    ax_box.set_ylabel(y)
    ax_box.set_title(title if title else f"{y} distribution",fontdict={"fontsize": 18})
    
    # Adjust layout to make room for bottom legend (only when hue is present)
    if hue is not None:
        fig.subplots_adjust(bottom=0.25)

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
    default_filter = {
        "carrier_freq":         12.0,
        # Peak shape in frequency domain
        "carrier_waveform":     "sine",   # "gaussian" | "sine" | "burst"
        # Frequency modulation
        "mod_amp":              0.0,
        "mod_freq":             0.0,
        # Aperiodic component
        "aperiodic_exponent":   2.0,          # β — slope of 1/f^β
        "has_aperiodic":        True,
        # Peak(s)
        "n_peaks":              1,
        "peak_bw":              0.5,          # Gaussian σ in Hz
        # KEY PARAM: peak power relative to aperiodic floor at aperiodic_ref_freq
        "peak_snr_db":          20.0,
        # Noise
        "noise_type":           "white",       # "None" | "white" | "pink"
        # Noise PSD relative to power at carrier_freq
        "noise_snr_db":         -30.0,
        # Analysis
        "window_length_sec":    5,
        "fft_method":          "fft",       # "fft" | "welch"
    }

    # Each plot loads only what it needs
    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "window_length_sec")),
        HDF_PATH,
        x="algorithm", y="abs_error", hue="window_length_sec",
        title="Effect of Window Length\n"
    )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "noise_type")),
        HDF_PATH,
        x="algorithm", y="abs_error", hue="noise_type",
        title="Effect of Noise Type\n"
    )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding({**default_filter.copy(),"noise_type": "white"}, "noise_snr_db")),
        HDF_PATH,
        x="algorithm", y="abs_error", hue="noise_snr_db",
        title="Effect of Noise Level\n"
    )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "carrier_freq")),
        HDF_PATH,
        x="algorithm", y="abs_error", hue="carrier_freq",
        title="Effect of Carrier Frequency\n"
    )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "carrier_waveform")),
        HDF_PATH,
        x="algorithm", y="abs_error", hue="carrier_waveform",
        title="Effect of Carrier Waveform\n"
    )

    # plot_box(
    #     load_samples_for_plot(HDF_PATH, df_metrics,
    #                           **params_excluding(default_filter, "mod_freq")),
    #     HDF_PATH,
    #     x="algorithm", y="abs_error", hue="mod_freq",
    #     title="Effect of Modulation Frequency\n"
    # )

    # plot_box(
    #     load_samples_for_plot(HDF_PATH, df_metrics,
    #                           **params_excluding({**default_filter.copy(), "mod_freq": 0.01}, "mod_amp")),
    #     HDF_PATH,
    #     x="algorithm", y="abs_error", hue="mod_amp",
    #     title="Effect of Modulation Amplitude\n"
    # )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "n_peaks")),
        HDF_PATH,
        x="algorithm", y="abs_error", hue="n_peaks",
        title="Effect of Number of Peaks\n"
    )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "has_aperiodic")),
        HDF_PATH,
        x="algorithm", y="abs_error", hue="has_aperiodic",
        title="Effect of Aperiodic Component\n"
    )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "aperiodic_exponent")),
        HDF_PATH,
        x="algorithm", y="abs_error", hue="aperiodic_exponent",
        title="Effect of Aperiodic Exponent\n"
    )

    plot_box(
        load_samples_for_plot(HDF_PATH, df_metrics,
                              **params_excluding(default_filter, "fft_method")),
        HDF_PATH,
        x="algorithm", y="abs_error", hue="fft_method",
        title="Effect of FFT method\n"
    )

    # # Print detection rates for specific conditions
    # print_detection_table(df_metrics, window_length_sec=5, noise_type="None", noise_snr_db=0, mod_freq=0, mod_amp=0.5, carrier_freq=10.0, 
    #                       n_peaks=0,has_aperiodic=True, carrier_waveform="delta")
    # print_detection_table(df_metrics, window_length_sec=5, noise_type="white", noise_snr_db=-5, mod_freq=0, mod_amp=0.5, carrier_freq=10.0, n_peaks=1,
    #                       has_aperiodic=True, carrier_waveform="delta")


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
    
    plot_box(load_samples_for_plot(HDF_PATH, df_metrics), HDF_PATH, x="algorithm", y="abs_error", title="MAE distribution across all conditions"),

    # # Print Conditions where MAE > 4Hz for any algorithm
    # high_error = df_metrics[df_metrics["mae"] > 4]
    # print("Conditions with MAE > 4Hz:")
    # print(high_error[["algorithm", "mae"] + [col for col in df_metrics.columns if col not in ['condition_id', 'algorithm', 'mae',"fs", "signal_length_sec", "freq_range", "alpha_band",
    #     "SG_window", "SG_poly", "pink_ax_r2"]]])

    plt.show()