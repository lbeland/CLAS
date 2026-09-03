"""Alpha-frequency (IAF) variability plots from ``run_pipeline.py`` output.

Reads ``iaf_per_segment.csv`` (columns: file, segment_index, had_alpha, paf_hz,
time_s) from one or two run directories and reports the per-participant
coefficient of variation (CV) of the IAF,

    CV_p = std(IAF_p) / mean(IAF_p)

over the sliding IAF windows of that participant's recording.

Usage
-----
    # single method
    python plot_results.py  RESULTS/ds004148_Fz_fooof

    # compare two methods (e.g. fooof vs simple)
    python plot_results.py  RESULTS/ds004148_Fz_fooof  RESULTS/ds004148_Fz_simple
    python plot_results.py  DIR_A DIR_B --labels fooof simple

Writes ``iaf_cv_per_file.csv`` (+ ``iaf_change_rate_per_file.csv`` in single
mode) into each run dir; figures go to ``MA/plots`` as ``.pgf`` + ``.pdf``
(repo convention, cf. ``analysis/plot.py`` / ``IAF_Estimation/plot_iaf_tests.py``).
"""

import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})
import matplotlib.pyplot as plt  # noqa: E402

_IAF_CSV = "iaf_per_segment.csv"
_PARTICIPANT_RE = r"(s\w+?)(?=_)"

# LaTeX symbol for the per-window IAF estimate (text.usetex is on).
_F0 = r"$\tilde{f}_0$"

PLOTS_DIR = "/home/linda/Documents/MA/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# figure sizing, matching analysis/plot.py
TEXTWIDTH = 6.30045
ASPECT_RATIO = 9 / 16
FIG_WIDTH = TEXTWIDTH
FIG_HEIGHT = FIG_WIDTH * ASPECT_RATIO
FIGSIZE = (FIG_WIDTH, FIG_HEIGHT)


def _tex(s):
    """Escape LaTeX-special chars in dynamic label text (usetex mode)."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def _save(fig, name):
    """Save ``<name>.pgf`` and ``<name>.pdf`` into ``MA/plots`` (repo convention)."""
    for ext in ("pgf", "pdf"):
        fig.savefig(os.path.join(PLOTS_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {PLOTS_DIR}/{name}.{{pgf,pdf}}")


# --------------------------------------------------------------------------
# data reduction
# --------------------------------------------------------------------------
def _valid(df):
    """Rows with a usable per-window IAF, tagged with a ``participant`` id."""
    if "snr" not in df.columns:          # csv from an older run / fooof method
        df = df.assign(snr=np.nan)
    m = (
        (df["segment_index"] >= 0)
        & (df["had_alpha"] == 1)
        & np.isfinite(df["paf_hz"])
    )
    d = df.loc[m].copy()
    d["participant"] = d["file"].str.extract(_PARTICIPANT_RE, expand=False)
    return d


def cv_per_participant(df):
    """mean / std / n / CV of the per-window IAF (+ mean SNR), one row per participant."""
    d = _valid(df)
    if d.empty:
        raise RuntimeError("no valid alpha segments in iaf_per_segment.csv")
    g = (
        d.groupby("participant")
        .agg(
            mean_iaf_hz=("paf_hz", "mean"),
            std_iaf_hz=("paf_hz", "std"),
            n_segments=("paf_hz", "count"),
            mean_snr=("snr", "mean"),
            median_snr=("snr", "median"),
        )
        .reset_index()
    )
    g["cv_iaf"] = g["std_iaf_hz"] / g["mean_iaf_hz"]
    return g


def change_rate_per_participant(df):
    """Rate of IAF change (|Δf|/Δt, Hz/s) between consecutive windows."""
    d = _valid(df)
    rows = []
    for pid, grp in d.groupby("participant"):
        grp = grp.sort_values("time_s")
        p = grp["paf_hz"].to_numpy()
        t = grp["time_s"].to_numpy()
        if len(p) < 2:
            continue
        rate = np.abs(np.diff(p)) / np.diff(t)
        rows.append(dict(
            participant=pid,
            mean_rate_hz_per_s=float(np.mean(rate)),
            median_rate_hz_per_s=float(np.median(rate)),
            std_rate_hz_per_s=float(np.std(rate)),
            n_windows=int(len(p)),
        ))
    return pd.DataFrame(rows)


def _describe_cv(cv, label=""):
    cv = np.asarray(cv, dtype=float)
    q1, med, q3 = np.percentile(cv, [25, 50, 75])
    sd = cv.std(ddof=1) if cv.size > 1 else float("nan")
    print(f"  CV [{label:6s}]  n={cv.size:3d}  "
          f"median={med:.4f} ({med * 100:.2f} %)  IQR=[{q1:.4f}, {q3:.4f}]  "
          f"mean={cv.mean():.4f} ± {sd:.4f}")
    return med


def _label_for(run_dir):
    """'ds004148_Fz-FCz_simple' -> 'simple'; fall back to the dir name."""
    parts = pathlib.Path(run_dir).name.split("_")
    for cand in ("fooof", "simple"):
        if cand in parts:
            return cand
    return pathlib.Path(run_dir).name


# --------------------------------------------------------------------------
# single method
# --------------------------------------------------------------------------
def plot_single(run_dir, label):
    run_dir = pathlib.Path(run_dir)
    df = pd.read_csv(run_dir / _IAF_CSV)

    cv = cv_per_participant(df)
    rate = change_rate_per_participant(df)
    cv.to_csv(run_dir / "iaf_cv_per_file.csv", index=False)
    rate.to_csv(run_dir / "iaf_change_rate_per_file.csv", index=False)

    cvp = cv["cv_iaf"].to_numpy() * 100.0
    snr = cv["mean_snr"].to_numpy()
    has_snr = np.isfinite(snr).any()

    print(f"\n[{label}]  {len(df)} window rows, {cv['participant'].nunique()} participants")
    _describe_cv(cv["cv_iaf"], label)
    if not rate.empty:
        print(f"  |Δf|/Δt   {rate['mean_rate_hz_per_s'].mean():.4f} Hz/s  "
              f"(mean over participants of the per-recording mean step-to-step change)")
    if has_snr:
        ok = np.isfinite(snr) & np.isfinite(cvp)
        r = np.corrcoef(snr[ok], cvp[ok])[0, 1] if ok.sum() > 2 else float("nan")
        print(f"  mean SNR  median={np.nanmedian(snr):.3f}   "
              f"corr(mean SNR, CV) = {r:+.3f}  (n={int(ok.sum())})")

    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)

    ax[0].hist(cvp, bins=20, edgecolor="black", alpha=0.85)
    ax[0].axvline(np.nanmedian(cvp), color="C1", lw=2,
                  label=f"median {np.nanmedian(cvp):.2f}" r"\,\%")
    ax[0].set_xlabel(r"CV (\%)")
    ax[0].set_ylabel("participants")
    ax[0].legend()

    # ax[1].scatter(cv["mean_iaf_hz"], cv["std_iaf_hz"], s=20, alpha=0.7)
    # ax[1].set_xlabel(f"mean {_F0} (Hz)")
    # ax[1].set_ylabel(f"{_F0} std (Hz)")

    if has_snr:
        ax[1].scatter(snr, cvp, s=20, alpha=0.7)
        ax[1].set_xlabel("mean SNR")
        ax[1].set_ylabel(r"CV (\%)")
    else:
        ax[1].text(0.5, 0.5, "no SNR\n(fooof method)", ha="center", va="center",
                   transform=ax[1].transAxes)
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    fig.tight_layout()
    _save(fig, f"iaf_cv_{label}")


# --------------------------------------------------------------------------
# comparison of two methods
# --------------------------------------------------------------------------
def plot_compare(dir_a, dir_b, labels):
    la, lb = labels
    dir_a, dir_b = pathlib.Path(dir_a), pathlib.Path(dir_b)
    da = pd.read_csv(dir_a / _IAF_CSV)
    db = pd.read_csv(dir_b / _IAF_CSV)

    # per-window agreement: match windows by (recording, window centre time)
    va, vb = _valid(da), _valid(db)
    for v in (va, vb):
        v["time_s"] = v["time_s"].round(6)
    m = va.merge(vb, on=["file", "time_s"], suffixes=(f"_{la}", f"_{lb}"))
    pa = m[f"paf_hz_{la}"].to_numpy()
    pb = m[f"paf_hz_{lb}"].to_numpy()
    diff = pa - pb
    md, sd = float(np.nanmean(diff)), float(np.nanstd(diff, ddof=1))
    loa = (md - 1.96 * sd, md + 1.96 * sd)

    # per-participant CV for each method
    cva = cv_per_participant(da)
    cvb = cv_per_participant(db)
    cva.to_csv(dir_a / "iaf_cv_per_file.csv", index=False)
    cvb.to_csv(dir_b / "iaf_cv_per_file.csv", index=False)
    cvm = cva.merge(cvb, on="participant", suffixes=(f"_{la}", f"_{lb}"))

    print(f"\n[compare]  {la} vs {lb}   ({len(m)} matched windows, "
          f"{cvm['participant'].nunique()} participants)")
    print(f"  per-window IAF  {la}-{lb}:  mean diff = {md:+.3f} Hz   "
          f"SD = {sd:.3f} Hz   95% LoA = [{loa[0]:+.3f}, {loa[1]:+.3f}] Hz")
    med_a = _describe_cv(cvm[f"cv_iaf_{la}"], la)
    med_b = _describe_cv(cvm[f"cv_iaf_{lb}"], lb)

    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)

    # (1) per-window difference histogram
    ax[0].hist(diff, bins=60, edgecolor="black")
    ax[0].axvline(md, color="C1", lw=2, label=f"mean {md:+.3f} Hz")
    ax[0].set_xlabel(f"per-window {_F0}: {_tex(la)} $-$ {_tex(lb)} (Hz)")
    ax[0].set_ylabel("windows")
    ax[0].legend(fontsize=8)

    # (2) per-participant CV, method vs method
    ca = cvm[f"cv_iaf_{la}"].to_numpy() * 100.0
    cb = cvm[f"cv_iaf_{lb}"].to_numpy() * 100.0
    ax[1].plot(ca, cb, "o", alpha=0.7)
    lim = [0, float(np.nanmax([ca, cb])) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlabel(rf"CV {_tex(la)} (\%), median {med_a * 100:.2f}")
    ax[1].set_ylabel(rf"CV {_tex(lb)} (\%), median {med_b * 100:.2f}")

    fig.tight_layout()
    _save(fig, f"iaf_cv_compare_{la}_vs_{lb}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=pathlib.Path, nargs="+",
                   help="one run dir (single-method CV) or two (method comparison)")
    p.add_argument("--labels", nargs=2, metavar=("A", "B"),
                   help="labels for the two run dirs (default: inferred from names)")
    args = p.parse_args()

    if len(args.run_dir) == 1:
        d = args.run_dir[0]
        plot_single(d, _label_for(d))
    elif len(args.run_dir) == 2:
        a, b = args.run_dir
        labels = args.labels or [_label_for(a), _label_for(b)]
        plot_compare(a, b, labels)
    else:
        p.error("give one or two run directories")
