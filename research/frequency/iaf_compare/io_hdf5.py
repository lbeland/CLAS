"""HDF5 result store: writing pooled sweep results, and loading them back
into tidy DataFrames / arrays for plotting.

Layout::

    conditions/<cond_idx:04d>/
        <param attrs...>
        spectrum_psd, spectrum_freq_bins, first_window, ground_truth
        <algo>/estimates, <algo>/errors, <algo>.attrs = pooled metrics

``spectrum_psd`` / ``spectrum_freq_bins`` is one example spectrum per condition
(first seed encountered) -- the Welch PSD, as chosen in
``pipeline.run_window_analysis`` (smoother than the raw periodogram for the
spectrum panels); ``first_window`` is that same seed's time-domain window.
"""
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from .metrics import compute_pooled_metrics
from .paths import RESULTS_H5

_NON_ALGO_KEYS = ("ground_truth", "first_window", "spectrum_psd", "spectrum_freq_bins")


# --- Writing ---------------------------------------------------------------

def write_results(all_results, out_path=RESULTS_H5):
    """Pool ``all_results`` (list of process_condition return tuples) and
    write the HDF5 store to ``out_path``.

    Pooling collapses every seed sharing the same condition (identical
    stored_config) into one entry: a single example spectrum/window (all seeds
    of a condition look alike) plus, per algo, the full list of per-seed
    estimates -- then :func:`compute_pooled_metrics` scores each algo over all
    its seeds at once."""
    out_path = _as_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pooled_samples = {}
    example_by_condition = {}
    gt_by_condition = {}
    for _, config, gt, estimates_per_algo, spectrum_info in all_results:
        condition_key = tuple(sorted(config.items(), key=lambda x: str(x)))
        algo_samples = pooled_samples.setdefault(condition_key, {})
        for algo, est_pf in estimates_per_algo.items():
            samples = algo_samples.setdefault(algo, {"est": [], "gt": []})
            samples["est"].append(float(est_pf))
            samples["gt"].append(gt)
        if spectrum_info is not None:  # only seed 0 carries the example arrays
            example_by_condition.setdefault(condition_key, spectrum_info)
        gt_by_condition.setdefault(condition_key, gt)

    with h5py.File(out_path, "w") as hf:
        hf.attrs["created"] = str(pd.Timestamp.now())
        conds_grp = hf.create_group("conditions")

        for cond_idx, (condition_key, algo_samples) in enumerate(tqdm(
                pooled_samples.items(), total=len(pooled_samples), desc="Writing HDF5")):
            grp = conds_grp.create_group(f"{cond_idx:04d}")

            for k, v in dict(condition_key).items():
                if isinstance(v, tuple):
                    grp.attrs[k] = list(v)
                else:
                    grp.attrs[k] = v if v is not None else "none"

            example = example_by_condition.get(condition_key)
            if example is None:
                raise RuntimeError(
                    f"No example spectrum for condition {dict(condition_key)} -- "
                    "seed 0 must be present in all_results (it carries the example "
                    "arrays; see process_condition).")
            (freq_bins, psd), first_window = example
            grp.create_dataset("spectrum_psd",       data=psd,          compression="gzip")
            grp.create_dataset("spectrum_freq_bins", data=freq_bins,    compression="gzip")
            grp.create_dataset("first_window",       data=first_window, compression="gzip")
            grp.create_dataset("ground_truth",       data=np.array([gt_by_condition[condition_key]]), compression="gzip")

            for algo, samples in algo_samples.items():
                metrics = compute_pooled_metrics(samples["est"], samples["gt"])
                algo_grp = grp.create_group(algo)
                algo_grp.create_dataset("estimates", data=metrics.pop("estimates"), compression="gzip")
                algo_grp.create_dataset("errors",    data=metrics.pop("errors"),    compression="gzip")
                for metric, val in metrics.items():
                    algo_grp.attrs[metric] = val
    return out_path


# --- Loading -------------------------------------------------------------

def params_excluding(base_params, exclude_key):
    """Copy of base_params with exclude_key removed (so it can be used as a
    plotting hue while everything else stays pinned)."""
    p = dict(base_params)
    p.pop(exclude_key, None)
    return p


def filter_df(df, **kwargs):
    """Filter DataFrame to rows matching all keyword conditions. A list/tuple
    value matches any of its entries."""
    mask = pd.Series(True, index=df.index)
    for col, val in kwargs.items():
        if isinstance(val, (list, tuple)):
            mask &= df[col].isin(val)
        else:
            mask &= (df[col] == val)
    return df[mask]


def load_metrics(path=RESULTS_H5):
    """Load just the aggregated metrics into a DataFrame."""
    rows = []
    with h5py.File(path, "r") as hf:
        for cond_id, grp in hf["conditions"].items():
            config = dict(grp.attrs)
            for algo, algo_grp in grp.items():
                if algo in _NON_ALGO_KEYS:
                    continue
                rows.append({
                    "condition_id": cond_id,
                    **config,
                    "algorithm": algo,
                    **dict(algo_grp.attrs)
                })
    return pd.DataFrame(rows)


def load_samples_for_plot(hdf_path, df_metrics, **filter_kwargs):
    """Filter conditions first, then load only matching per-seed samples from
    HDF5 (one row per seed; nan where a seed had no estimate/error)."""
    df_filtered = filter_df(df_metrics, **filter_kwargs)

    rows = []
    with h5py.File(hdf_path, "r") as hf:
        for _, row in df_filtered.iterrows():
            cond_id = f"{int(row['condition_id']):04d}"
            algo = row["algorithm"]
            try:
                errors = hf[f"conditions/{cond_id}/{algo}/errors"][:]
                estimates = hf[f"conditions/{cond_id}/{algo}/estimates"][:]
                row_dict = row.to_dict()
                for err, est in zip(errors, estimates):
                    rows.append({**row_dict, "error": err,
                                 "abs_error": abs(err), "estimate": est})
            except KeyError:
                pass
    return pd.DataFrame(rows)


def load_spectra_by_hue(hdf_path, df_filtered, hue):
    """First found spectrum per unique hue value. dict: hue_value -> (freq_bins, psd)."""
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
    """First found example window per unique hue value. dict: hue_value -> window."""
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


def _as_path(p):
    from pathlib import Path
    return p if isinstance(p, Path) else Path(p)
