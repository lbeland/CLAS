"""Alpha-frequency (IAF) variability plots from ``run_pipeline.py`` output.

Reads ``iaf_per_segment.csv`` (columns: file, segment_index, had_alpha, paf_hz,
time_s) from one or two run directories and reports the per-participant
standard deviation of the IAF,

    SD_p = std(IAF_p)   [Hz]

over the sliding IAF windows of that participant's recording. The coefficient
of variation (CV_p = SD_p / mean(IAF_p)) is still computed and written to the
per-file csv, but the figures now use the plain SD.

Usage
-----
    # single method
    python plot_results.py  RESULTS/ds004148_Fz_fooof

    # compare two methods (e.g. fooof vs simple)
    python plot_results.py  RESULTS/ds004148_Fz_fooof  RESULTS/ds004148_Fz_simple
    python plot_results.py  DIR_A DIR_B --labels fooof simple

Writes ``iaf_stats_per_file.csv`` (+ ``iaf_change_rate_per_file.csv`` in single
mode) into each run dir; figures go to ``MA/plots`` as ``.pgf`` + ``.pdf``
(repo convention, cf. ``analysis/plot.py`` / ``IAF_Estimation/plot_iaf_tests.py``).
"""

import argparse
import os
import pathlib

import numpy as np
import pandas as pd
from scipy import stats
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
    # No bbox_inches="tight": it re-crops the canvas to the artists and the
    # saved figure ends up narrower than FIG_WIDTH, so it won't fill \linewidth
    # when \input into the thesis. Rely on fig.tight_layout() instead.
    for ext in ("pgf", "pdf"):
        fig.savefig(os.path.join(PLOTS_DIR, f"{name}.{ext}"))
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


def stats_per_participant(df):
    """mean / std / n / CV of the per-window IAF (+ mean SNR), one row per participant.

    ``std_iaf_hz`` (Hz) is the statistic the figures use; ``cv_iaf`` is kept in
    the table for reference but no longer plotted.
    """
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
    """Rate of IAF change (|Δf|/Δt, Hz/s) between consecutive windows.

    Step-to-step differences are taken *within* each recording and only then
    pooled per participant. Diffing across the pooled sessions would be wrong:
    every ds004148 session shares the same ~0-300 s time axis, so sorting the
    pooled rows by ``time_s`` interleaves the sessions, producing spurious
    ``|Δf|`` jumps and ``Δt <= 0`` (division by zero) at every boundary.
    """
    d = _valid(df)
    rows = []
    for pid, grp in d.groupby("participant"):
        rates = []
        n_windows = 0
        n_recordings = 0
        for _, rec in grp.groupby("file"):
            rec = rec.sort_values("time_s")
            p = rec["paf_hz"].to_numpy()
            t = rec["time_s"].to_numpy()
            n_windows += len(p)
            if len(p) < 2:
                continue
            dt = np.diff(t)
            ok = dt > 0
            if not ok.any():
                continue
            rates.append(np.abs(np.diff(p))[ok] / dt[ok])
            n_recordings += 1
        if not rates:
            continue
        rate = np.concatenate(rates)
        rows.append(dict(
            participant=pid,
            mean_rate_hz_per_s=float(np.mean(rate)),
            median_rate_hz_per_s=float(np.median(rate)),
            std_rate_hz_per_s=float(np.std(rate)),
            n_windows=int(n_windows),
            n_recordings=int(n_recordings),
        ))
    return pd.DataFrame(rows)


def _describe_std(sd, label=""):
    """Summarise the per-participant IAF SD; participants with a single pooled
    window (SD undefined) are dropped from the summary."""
    sd = np.asarray(sd, dtype=float)
    sd = sd[np.isfinite(sd)]
    q1, med, q3 = np.percentile(sd, [25, 50, 75])
    spread = sd.std(ddof=1) if sd.size > 1 else float("nan")
    print(f"  SD [{label:6s}]  n={sd.size:3d}  "
          f"median={med:.4f} Hz  IQR=[{q1:.4f}, {q3:.4f}] Hz  "
          f"mean={sd.mean():.4f} ± {spread:.4f} Hz")
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

    st = stats_per_participant(df)
    rate = change_rate_per_participant(df)
    st.to_csv(run_dir / "iaf_stats_per_file.csv", index=False)
    rate.to_csv(run_dir / "iaf_change_rate_per_file.csv", index=False)

    sdp = st["std_iaf_hz"].to_numpy()
    n_single = int(np.isnan(sdp).sum())
    if n_single:
        print(f"note: {n_single} participant(s) with a single pooled window "
              f"(SD undefined) dropped from SD figures/summary")
    print(f"Min SD: {np.nanmin(sdp):.4f} Hz  Max SD: {np.nanmax(sdp):.4f} Hz  ")
    snr = st["mean_snr"].to_numpy()
    has_snr = np.isfinite(snr).any()

    print(f"\n[{label}]  {len(df)} window rows, {st['participant'].nunique()} participants")
    _describe_std(st["std_iaf_hz"], label)
    if not rate.empty:
        print(f"  |Δf|/Δt   {rate['mean_rate_hz_per_s'].mean():.4f} Hz/s  "
              f"(mean over participants of the per-recording mean step-to-step change)")
        print(f"  |Δf|/Δt   {rate['mean_rate_hz_per_s'].std():.4f} Hz/s  "
              f"(standard deviation over participants of the per-recording mean step-to-step change)")
    if has_snr:
        ok = np.isfinite(snr) & np.isfinite(sdp)
        if ok.sum() > 2:
            rho, pval = stats.spearmanr(snr[ok], sdp[ok])
            print(f"  mean SNR  median={np.nanmedian(snr):.3f}")
            print(f"  Spearman(mean SNR, SD):  r = {rho:+.3f}   "
                  f"p = {pval:.3g}   (n = {int(ok.sum())})")
        else:
            print(f"  mean SNR  median={np.nanmedian(snr):.3f}   "
                  f"Spearman(mean SNR, SD): n too small (n={int(ok.sum())})")
        top = st.loc[st["mean_snr"].idxmax()]
        print(f"  Highest mean SNR: {top['participant']} "
              f"({top['mean_snr']:.3f})")
        bottom = st.loc[st["mean_snr"].idxmin()]
        print(f"  Lowest mean SNR: {bottom['participant']} "
              f"({bottom['mean_snr']:.3f})")
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
    ax[0].set_title("(a)")
    ax[1].set_title("(b)")

    sdp_ok = sdp[np.isfinite(sdp)]
    ax[0].hist(sdp_ok, bins=20, edgecolor="black", alpha=0.85)
    ax[0].axvline(sdp_ok.mean(), color="grey", lw=2, linestyle="--",
                  label=f"mean {sdp_ok.mean():.3f}" r"\,Hz")
    ax[0].set_xlabel(f"{_F0} std [Hz]")
    ax[0].set_ylabel("participants")
    ax[0].legend()

    if has_snr:
        ax[1].scatter(snr, sdp, s=20, alpha=0.7)
        ax[1].set_xlabel("mean SNR")
        ax[1].set_ylabel(f"{_F0} std [Hz]")
    else:
        ax[1].text(0.5, 0.5, "no SNR\n(fooof method)", ha="center", va="center",
                   transform=ax[1].transAxes)
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    fig.tight_layout()
    _save(fig, f"iaf_stats_{label}")

    # std vs mean IAF, one point per participant
    fig2, ax2 = plt.subplots(figsize=FIGSIZE)

    mx = st["mean_iaf_hz"].to_numpy()
    sy = st["std_iaf_hz"].to_numpy()
    ok2 = np.isfinite(mx) & np.isfinite(sy)
    ax2.scatter(mx[ok2], sy[ok2], s=20, alpha=0.7)
    lr = stats.linregress(mx[ok2], sy[ok2])
    xx = np.linspace(mx[ok2].min(), mx[ok2].max(), 100)
    ax2.plot(xx, lr.intercept + lr.slope * xx, color="grey", lw=2, linestyle="--",
             label=rf"$r = {lr.rvalue:+.3f}$, $p = {lr.pvalue:.3g}$")
    ax2.set_xlabel(f"mean {_F0} [Hz]")
    ax2.set_ylabel(f"{_F0} std [Hz]")
    ax2.legend()
    fig2.tight_layout()
    _save(fig2, f"iaf_std_vs_mean_{label}")

    # per-participant distribution of the per-window IAF: one translucent KDE
    # curve per participant over a common frequency axis (all recordings pooled)
    d = _valid(df)
    vals_all = d["paf_hz"].to_numpy()
    lo, hi = np.percentile(vals_all, [0.5, 99.5])
    xs = np.linspace(lo - 0.5, hi + 0.5, 400)

    fig3, ax3 = plt.subplots(figsize=FIGSIZE)
    n_curves = n_skipped = 0
    for _, grp in d.groupby("participant"):
        v = grp["paf_hz"].to_numpy()
        if v.size < 3 or np.ptp(v) == 0:      # KDE needs a bit of spread
            n_skipped += 1
            continue
        kde = stats.gaussian_kde(v)
        ax3.plot(xs, kde(xs), color="C0", lw=1, alpha=0.25)
        # mu = float(np.mean(v))
        # ax3.scatter(mu, kde(mu), s=12, color="C1", alpha=0.6, zorder=3)
        n_curves += 1
    # ax3.plot(xs, stats.gaussian_kde(vals_all)(xs), color="black", lw=1.5,
    #          label=f"all participants pooled (n={n_curves})")
    ax3.set_xlabel(f"{_F0} [Hz]")
    ax3.set_ylabel("density")
    ax3.set_xlim(xs[0], xs[-1])
    # ax3.legend()
    fig3.tight_layout()
    _save(fig3, f"iaf_dist_{label}")
    if n_skipped:
        print(f"note: {n_skipped} participant(s) with < 3 pooled windows "
              f"omitted from the per-participant distribution plot")


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

    # per-participant IAF stats for each method
    sta = stats_per_participant(da)
    stb = stats_per_participant(db)
    sta.to_csv(dir_a / "iaf_stats_per_file.csv", index=False)
    stb.to_csv(dir_b / "iaf_stats_per_file.csv", index=False)
    stm = sta.merge(stb, on="participant", suffixes=(f"_{la}", f"_{lb}"))

    print(f"\n[compare]  {la} vs {lb}   ({len(m)} matched windows, "
          f"{stm['participant'].nunique()} participants)")
    print(f"  per-window IAF  {la}-{lb}:  mean diff = {md:+.3f} Hz   "
          f"SD = {sd:.3f} Hz   95% LoA = [{loa[0]:+.3f}, {loa[1]:+.3f}] Hz")
    med_a = _describe_std(stm[f"std_iaf_hz_{la}"], la)
    med_b = _describe_std(stm[f"std_iaf_hz_{lb}"], lb)

    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, width_ratios=[1.5, 1])
    ax[0].set_title("(a)")
    ax[1].set_title("(b)")

    # (1) per-window difference histogram
    ax[0].hist(diff, bins=60, edgecolor="black")
    ax[0].axvline(md, color="grey", lw=2, linestyle="--", label=f"mean {md:+.3f} Hz")
    ax[0].set_xlabel(f"per-window {_F0}: {_tex(la)} $-$ {_tex(lb)} (Hz)")
    ax[0].set_ylabel("windows")
    ax[0].legend(fontsize=8)

    # (2) per-participant IAF std, method vs method
    ca = stm[f"std_iaf_hz_{la}"].to_numpy()
    cb = stm[f"std_iaf_hz_{lb}"].to_numpy()
    ax[1].plot(ca, cb, "o", alpha=0.7)
    lim = [0, float(np.nanmax([ca, cb])) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlabel(rf"{_F0} std {_tex(la)} [Hz]")
    ax[1].set_ylabel(rf"{_F0} std {_tex(lb)} [Hz]")

    fig.tight_layout()
    _save(fig, f"iaf_stats_compare_{la}_vs_{lb}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=pathlib.Path, nargs="+",
                   help="one run dir (single-method IAF stats) or two (method comparison)")
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
