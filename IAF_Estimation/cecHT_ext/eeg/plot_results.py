"""Three-panel EEG phase-estimate figure (CLAS overlay).

Same figure as upstream ``EEG/eeg_plot.py`` (polar A/B + phase-error-vs-IAF-CV C)
with local tweaks:
- participant id regex ``(s\\w+?)(?=_)`` (matches the ds004148 ``sub-XX`` ids),
- also writes ``iaf_cv_per_file.csv`` and ``iaf_change_rate_per_file.csv``,
- ``save_base`` defaults to ``phase_error_ds004148``.

Reads ``phase_error_all.npz`` / ``phase_error_per_file.csv`` /
``iaf_per_segment.csv`` from a run directory (default: current directory; pass a
path as the first argument, e.g. a ``cecHT_ext/results/<...>/`` folder produced by
``run_pipeline.py``) and writes the figure + CSVs back into it.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401  -> puts the cecHT submodule on sys.path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import make_figure  # noqa: E402

NPZ_PATH = "phase_error_all.npz"
PHASE_CSV = "phase_error_per_file.csv"
IAF_CSV = "iaf_per_segment.csv"

_PARTICIPANT_RE = r"(s\w+?)(?=_)"


# --------------------------------------------------------------------------
# IAF helpers
# --------------------------------------------------------------------------
def compute_iaf_cv_per_file(iaf_csv: str):
    df = pd.read_csv(iaf_csv)

    mask_valid = (
        (df["segment_index"] >= 0)
        & (df["had_alpha"] == 1)
        & np.isfinite(df["paf_hz"])
    )
    df_valid = df.loc[mask_valid].copy()
    if df_valid.empty:
        raise RuntimeError("No valid segments with alpha found")

    df_valid["participant"] = df_valid["file"].str.extract(_PARTICIPANT_RE, expand=False)

    stats = (
        df_valid
        .groupby("participant")["paf_hz"]
        .agg(mean_iaf_hz="mean", std_iaf_hz="std", n_segments="count")
        .reset_index()
        .rename(columns={"participant": "file"})
    )
    stats["cv_iaf"] = stats["std_iaf_hz"] / stats["mean_iaf_hz"]
    return stats


def compute_iaf_change_rate(iaf_csv: str):
    """Per-recording rate of IAF change (Hz/s) between consecutive windows."""
    df = pd.read_csv(iaf_csv)

    mask_valid = (
        (df["segment_index"] >= 0)
        & (df["had_alpha"] == 1)
        & np.isfinite(df["paf_hz"])
    )
    df_valid = df.loc[mask_valid].copy()

    rows = []
    for file, grp in df_valid.groupby("file"):
        grp = grp.sort_values("time_s")
        pafs = grp["paf_hz"].values
        times = grp["time_s"].values
        if len(pafs) < 2:
            continue
        dt = np.diff(times)
        dp = np.abs(np.diff(pafs))
        rate = dp / dt
        rows.append(dict(
            file=file,
            mean_rate_hz_per_s=np.mean(rate),
            median_rate_hz_per_s=np.median(rate),
            std_rate_hz_per_s=np.std(rate),
            n_windows=len(pafs),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def plot_phase_error(
    phase_err_unc_deg_all,
    phase_err_cal_deg_all,
    phase_csv_path=PHASE_CSV,
    iaf_csv_path=IAF_CSV,
    save_base="phase_error_ds004148",
):
    phase_err_unc_rad = np.radians(np.asarray(phase_err_unc_deg_all))
    phase_err_cal_rad = np.radians(np.asarray(phase_err_cal_deg_all))

    phase_df = pd.read_csv(phase_csv_path)
    iaf_stats = compute_iaf_cv_per_file(iaf_csv_path)
    iaf_stats.to_csv("iaf_cv_per_file.csv", index=False)

    iaf_change_rate = compute_iaf_change_rate(iaf_csv_path)
    print(iaf_change_rate.describe())
    iaf_change_rate.to_csv("iaf_change_rate_per_file.csv", index=False)

    phase_df["participant"] = phase_df["file"].str.extract(_PARTICIPANT_RE, expand=False)

    phase_subject = phase_df.groupby("participant").agg({
        "mean_unc_deg": "mean",
        "mean_cal_deg": "mean",
        "n_samples": "sum",
    }).reset_index()

    trial_mu_unc = np.radians(phase_subject["mean_unc_deg"].to_numpy(dtype=float))
    trial_mu_cal = np.radians(phase_subject["mean_cal_deg"].to_numpy(dtype=float))
    trial_nwin = phase_subject["n_samples"].to_numpy(dtype=float)

    merged = phase_subject.merge(iaf_stats, left_on="participant", right_on="file", how="left")
    if merged.empty:
        raise RuntimeError("No overlapping recordings after aggregating to subject level.")

    x_cv = merged["cv_iaf"].values
    y_unc = merged["mean_unc_deg"].values
    y_cal = merged["mean_cal_deg"].values

    make_figure(
        err_unc_rad=phase_err_unc_rad,
        err_cal_rad=phase_err_cal_rad,
        trial_freq_cv=x_cv,
        trial_abs_unc_rad=np.radians(y_unc),
        trial_abs_cal_rad=np.radians(y_cal),
        trial_mu_unc=trial_mu_unc,
        trial_mu_cal=trial_mu_cal,
        trial_nwin=trial_nwin,
        save_base=save_base,
        n_perm=int(1e5),
        perm_seed=0,
        panel_c_xlabel="IAF CV",
        panel_c_title=r"$\mathbf{c}$ Phase error vs. IAF variability",
    )


def main(npz_path=NPZ_PATH, phase_csv_path=PHASE_CSV, iaf_csv_path=IAF_CSV):
    data = np.load(npz_path)
    if "phase_err_unc_deg_all" not in data.files or "phase_err_cal_deg_all" not in data.files:
        raise KeyError(
            "NPZ file must contain 'phase_err_unc_deg_all' and 'phase_err_cal_deg_all'."
        )

    phase_err_unc_deg_all = data["phase_err_unc_deg_all"]
    phase_err_cal_deg_all = data["phase_err_cal_deg_all"]
    print(f"Loaded {phase_err_unc_deg_all.size} uncalibrated samples "
          f"and {phase_err_cal_deg_all.size} calibrated samples from {npz_path}")

    plot_phase_error(
        phase_err_unc_deg_all,
        phase_err_cal_deg_all,
        phase_csv_path=phase_csv_path,
        iaf_csv_path=iaf_csv_path,
    )


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="Plot phase-error results from a run directory.")
    p.add_argument("dir", nargs="?", default=".",
                   help="Run directory holding phase_error_all.npz etc. (default: .).")
    args = p.parse_args()
    os.chdir(args.dir)
    main()
