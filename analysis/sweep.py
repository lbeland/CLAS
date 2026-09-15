"""
SNR / noise-colour sweep analysis: turn a folder of CLAS runs (one per
(snr_db, noise_color) point, as produced by sweep_clas.py) into either a
phase-error-vs-SNR figure (analyse_snr_sweep) or a summary error table
(snr_sweep_table).

Kept out of analysis/main.py -- which stays focused on the single-recording
analysis pipeline -- but built on its general helpers (compute_recording_errors,
find_result_dirs).

Run from the repo root, either as a module or as a script:

    python -m analysis.sweep          # runs the __main__ block below
    python analysis/sweep.py

or from your own code / a REPL:

    from analysis.sweep import snr_sweep_table
    snr_sweep_table("results/snr_sweep/ecHTtests", kind="echt")
"""

import csv
import glob
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd

# Allow `python analysis/sweep.py` (not just `python -m analysis.sweep`): put
# the repo root on sys.path so `import analysis.*` resolves. Same hack as main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.main import compute_recording_errors, find_result_dirs
from analysis.plot import PLOTS_DIR

# Error-series labels emitted by analysis.core.compute_errors(). Kept here as
# constants so this module tracks any rename there in one place.
ONLINE_VS_GT      = r"$\hat \theta - \theta$"            # online estimate - ground truth
HILBERT_VS_GT     = r"$\theta_{\mathrm{HT}} - \theta$"   # offline Hilbert - ground truth
ONLINE_VS_HILBERT = r"$\hat\theta - \theta_{\mathrm{HT}}$"  # online estimate - offline Hilbert


def _phase_error_stats(values: np.ndarray) -> "tuple[float, float, int]":
    """Circular mean and circular SD (both in degrees) of a wrapped phase-error
    series, after dropping non-finite samples.

    Series derived from the offline Hilbert phase (HILBERT_VS_GT,
    ONLINE_VS_HILBERT) already carry NaN on their own leading/trailing edges
    (hilbert_phase is NaN-masked at its own edges in
    analysis.main.load_and_analyse), so no separate time-window trim is
    needed here -- and one that isn't Hilbert-derived (ONLINE_VS_GT, the
    online estimate scored directly against true_phase) is used in full
    instead of being needlessly cut at both ends. Circular statistics
    (scipy.stats.circmean / circstd) are used because the error is an angle:
    the mean stays a meaningful lead/lag bias even for a broad distribution,
    and the SD isn't inflated by the (-180, 180] wrap. These numbers match
    the polar-plot annotations in plot_errors()."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, 0
    phi  = np.radians(vals)
    mean = float(np.degrees(circmean(phi, high=np.pi, low=-np.pi)))
    std  = float(np.degrees(circstd(phi,  high=np.pi, low=-np.pi)))
    return mean, std, vals.size


def _phase_correlation(a_vals: np.ndarray, b_vals: np.ndarray) -> "tuple[float, int]":
    """Jammalamadaka-Sarma circular correlation between two wrapped
    phase-error series -- here the online estimate's error vs. ground truth
    (theta_hat - theta) and the offline Hilbert error vs. ground truth
    (theta_HT - theta).

    Both series are first centred on their own circular mean and the sine of
    each residual is taken; the correlation is the mean product of those
    sines normalised by the two sine RMS values:
    rho_JS = E[sin(a-mu_a) sin(b-mu_b)] / sqrt(E[sin^2(a-mu_a)] E[sin^2(b-mu_b)]).
    Ground truth is common to both series, so this isolates shared *error*
    structure: a positive value means the online and offline estimates tend
    to lead / lag the true phase together in this run, near zero means their
    departures from truth are unrelated.

    Only non-finite samples are dropped (see _phase_error_stats() for why no
    time-window trim is needed): b_vals is the Hilbert-derived series here,
    so its NaN edges alone already exclude those samples from both series via
    the shared `keep` mask. Returns (corr, n)."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size == 0:
        return np.nan, 0
    ra, rb = np.radians(a), np.radians(b)
    da = np.sin(ra - circmean(ra, high=np.pi, low=-np.pi))
    db = np.sin(rb - circmean(rb, high=np.pi, low=-np.pi))
    denom = float(np.sqrt(np.mean(da ** 2) * np.mean(db ** 2)))
    corr = float(np.mean(da * db) / denom) if denom > 0 else np.nan
    return corr, a.size


def _run_carrier_f0(opts: dict, fallback: float) -> float:
    """The known carrier frequency of a synthetic run
    (SimulatedSource.options.carrier_frequency), or `fallback` if unset."""
    try:
        return float(opts["carrier_frequency"])
    except (KeyError, TypeError, ValueError):
        return fallback

# Per-kind table layout: (spec, plain-text header, LaTeX header, per-value format).
# `spec` is a row key, or a (mean_key, std_key) pair rendered as "mean +/- std"
# ("$mean \\pm std$" in LaTeX).
_SWEEP_TABLE_COLUMNS = {
    "echt": [
        ("noise",                                 "noise",               "Noise",                              "{}"),
        ("snr_db",                                "SNR (dB)",             "SNR [dB]",                           "{:g}"),
        (("err_mean", "err_std"),                 "phase err (deg)",      r"$\hat\theta - \theta\,  [\degree]$",             "{:.2f}"),
        (("hilb_vs_gt_mean", "hilb_vs_gt_std"),   "Hilbert-vs-GT (deg)",  r"$\theta_{\text{HT}} - \theta\,  [\degree]$",     "{:.2f}"),
        (("vs_hilbert", "vs_hilbert_std"),        "vs-Hilbert (deg)",     r"$\hat\theta - \theta_{\text{HT}}\,  [\degree]$", "{:.2f}"),
        ("corr_online_hilb",                      "corr(on,HT)",          r"$\rho_{JS}$",                          "{:.3f}"),
    ],
}


_SWEEP_TABLE_CAPTIONS = {
    "echt": (r"Online cecHT phase estimation accuracy over the in-band SNR "
             r"$\times$ noise-colour sweep. Each cell is circular mean $\pm$ "
             r"circular SD (degrees) of the signed phase error over the run, for "
             r"the online estimate against ground truth ($\hat\theta - \theta$), "
             r"the online estimate against the offline Hilbert phase "
             r"($\hat\theta - \theta_{\text{HT}}$), and the Hilbert phase "
             r"against ground truth ($\theta_{\text{HT}} - \theta$). A non-zero "
             r"mean indicates a systematic phase lead ($>0$) or lag ($<0$). The "
             r"last column is the Jammalamadaka--Sarma circular correlation "
             r"$\rho_{JS}$ between the online and offline Hilbert errors vs.\ "
             r"ground truth, i.e.\ how much the two estimators depart from the "
             r"true phase together. Columns involving the offline Hilbert "
             r"phase have the first and last 10\% of each run excluded as "
             r"filter edge transients; the online-vs-ground-truth column "
             r"does not."),
}


def snr_sweep_table(sweep_dir: str, kind: str = "echt", f0: float = 10,
                    out_dir: str = None,
                    f0_is_truth: bool = True, whiten: bool = False,
                    caption: str = None, label: str = None) -> "list[dict]":
    """Summarise an SNR / noise-colour sweep as a table instead of a plot.

    One row per run folder under sweep_dir, keyed by that run's SimulatedSource
    snr_db / noise_color: circular mean phase error +/- circular SD (deg) from
    the online-vs-ground-truth phase-error series (online ecHT vs. true phase),
    plus the same for the online-vs-Hilbert and Hilbert-vs-GT errors. Errors
    carry the (estimate - reference) sign, i.e. theta_hat - theta, and match
    plot_errors()'s polar annotations. One extra column gives the circular
    correlation between the online and offline Hilbert errors vs. ground
    truth, per noise type and level -- a measure of how much the two phase
    estimators depart from the true phase together.

    f0_is_truth (default True): take each run's known carrier_frequency as the
    pre-Hilbert bandpass centre instead of re-estimating f0 offline (see
    analysis.main.load_and_analyse). Pass False to fall back to offline
    estimation with `f0` as the default. whiten (default False) leaves the 1/f
    aperiodic whitening of the offline reference off; pass whiten=True to
    compare against the whitened reference instead.

    The offline-Hilbert-derived columns (vs_hilbert*, hilb_vs_gt*, and
    corr_online_hilb) already exclude their own leading/trailing edge
    transients -- hilbert_phase is NaN-masked at its own edges in
    analysis.main.load_and_analyse, so those NaNs simply drop out of the
    circular stats below. The online-vs-ground-truth column (err_mean/std)
    doesn't touch the Hilbert phase at all and is computed over the full run.

    Prints the table and writes snr_sweep_<kind>.csv / .tex into out_dir
    (default: the shared plots directory). The .tex is a full centred table
    float with caption and label (\\ref-able as tab:snr_sweep_<kind> unless
    `label` overrides it; `caption` overrides the default wording). Returns
    the row dicts.
    """
    if kind not in _SWEEP_TABLE_COLUMNS:
        raise ValueError(f"kind must be one of {list(_SWEEP_TABLE_COLUMNS)}, got {kind!r}")
    out_dir  = out_dir or PLOTS_DIR
    run_dirs = find_result_dirs(sweep_dir)
    print(f"Found {len(run_dirs)} run folder(s) under {sweep_dir!r}.")

    rows: list[dict] = []
    for i, results_dir in enumerate(run_dirs, 1):
        print(f"\n{'=' * 80}\n[{i}/{len(run_dirs)}] {results_dir}\n{'=' * 80}")
        try:
            graph_file = glob.glob(os.path.join(results_dir, "*.yaml"))[0]
            with open(graph_file) as fh:
                procs = yaml.safe_load(fh)["graph"]["processors"]
            opts       = procs["SimulatedSource"]["options"]
            snr_db     = float(opts["snr_db"])
            noise_type = str(opts.get("noise_color", "white"))

            run_f0 = _run_carrier_f0(opts, f0) if f0_is_truth else f0
            errors = compute_recording_errors(run_f0, results_dir,
                                              f0_is_truth=f0_is_truth, whiten=whiten)

            series = next((e for e in errors if e["label"] == ONLINE_VS_GT), None)
            if series is None:
                print(f"  No {ONLINE_VS_GT!r} series; skipping.")
                continue
            e_mean, e_std, n = _phase_error_stats(series["values"])
            if n == 0:
                print("  No finite phase-error samples; skipping.")
                continue
            cmp = next((e for e in errors if e["label"] == ONLINE_VS_HILBERT), None)
            vh_mean, vh_std, _ = (_phase_error_stats(cmp["values"])
                                  if cmp is not None else (np.nan, np.nan, 0))
            hgt = next((e for e in errors if e["label"] == HILBERT_VS_GT), None)
            hgt_mean, hgt_std, _ = (_phase_error_stats(hgt["values"])
                                    if hgt is not None else (np.nan, np.nan, 0))
            # Circular correlation between the online estimate's error vs.
            # ground truth and the offline Hilbert error vs. ground truth
            # (shared truth cancels, so this is shared *error* structure).
            corr_oh, _ = (_phase_correlation(series["values"], hgt["values"])
                         if hgt is not None else (np.nan, 0))
            # compute_errors() already uses the (estimate - reference) sign,
            # i.e. these means are circmean(theta_hat - theta) directly.
            rows.append({"noise": noise_type, "snr_db": snr_db,
                         "err_mean": e_mean, "err_std": e_std,
                         "vs_hilbert": vh_mean, "vs_hilbert_std": vh_std,
                         "hilb_vs_gt_mean": hgt_mean, "hilb_vs_gt_std": hgt_std,
                         "corr_online_hilb": corr_oh})
            print(f"  {noise_type}, SNR {snr_db:g} dB: phase err "
                  f"{e_mean:.2f} +/- {e_std:.2f} deg; "
                  f"online-vs-Hilbert {vh_mean:.2f} +/- {vh_std:.2f} deg; "
                  f"Hilbert-vs-GT {hgt_mean:.2f} +/- {hgt_std:.2f} deg; "
                  f"corr(online,Hilbert) {corr_oh:.3f}")
        except Exception:
            print(f"FAILED: {results_dir}")
            traceback.print_exc()

    if not rows:
        print("No sweep data collected; nothing to tabulate.")
        return []

    rows.sort(key=lambda r: (r["noise"], r["snr_db"]))
    cols = _SWEEP_TABLE_COLUMNS[kind]

    def _num(fmt, val):
        if isinstance(val, float) and not np.isfinite(val):
            return "--"
        if isinstance(val, float) and val == 0.0:
            val = 0.0  # normalise -0.0 so it doesn't print as "-0.00"
        try:
            out = fmt.format(val)
        except (ValueError, TypeError):
            return str(val)
        try:  # a value that rounds to zero can still carry a leading "-"; drop it
            if out.startswith("-") and float(out) == 0.0:
                out = out[1:]
        except ValueError:
            pass
        return out

    def _cell(col, row, latex=False):
        spec, _text, _tex, fmt = col
        if isinstance(spec, tuple):  # (mean_key, std_key) -> "mean +/- std"
            m, s = _num(fmt, row.get(spec[0], np.nan)), _num(fmt, row.get(spec[1], np.nan))
            return rf"${m} \pm {s}$" if latex else f"{m} ± {s}"
        return _num(fmt, row.get(spec, np.nan))

    text_cells = [[_cell(c, r) for c in cols] for r in rows]
    widths     = [max(len(c[1]), *(len(tc[j]) for tc in text_cells)) for j, c in enumerate(cols)]

    def _line(cells):
        return "  ".join(cell.ljust(widths[j]) if j == 0 else cell.rjust(widths[j])
                         for j, cell in enumerate(cells))

    print(f"\nSNR sweep summary ({kind}) -- {sweep_dir}")
    print(_line([c[1] for c in cols]))
    print("  ".join("-" * w for w in widths))
    for tc in text_cells:
        print(_line(tc))

    os.makedirs(out_dir, exist_ok=True)
    stem = f"snr_sweep_{kind}"

    csv_path = os.path.join(out_dir, f"{stem}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([c[1] for c in cols])
        writer.writerows(text_cells)

    # LaTeX: one \multirow noise label per group of same-noise rows (rows are
    # pre-sorted by (noise, snr_db), so each group is contiguous), \midrule
    # between groups.
    latex_cells = [[_cell(c, r, latex=True) for c in cols] for r in rows]
    align = "l" + "r" * (len(cols) - 1)
    body  = [r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(c[2] for c in cols) + r" \\", r"\midrule"]
    i, first_group = 0, True
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j]["noise"] == rows[i]["noise"]:
            j += 1
        if not first_group:
            body.append(r"\midrule")
        first_group = False
        span = j - i
        for k, tc in enumerate(latex_cells[i:j]):
            first_cell = rf"\multirow{{{span}}}{{*}}{{{tc[0]}}}" if k == 0 else ""
            body.append(" & ".join([first_cell, *tc[1:]]) + r" \\")
        i = j
    body += [r"\bottomrule", r"\end{tabular}"]

    caption = caption or _SWEEP_TABLE_CAPTIONS[kind]
    label   = label or f"tab:snr_sweep_{kind}"
    tex = [r"\begin{table}[htbp]",
           r"  \centering",
           rf"  \caption{{{caption}}}",
           rf"  \label{{{label}}}",
           *("  " + ln for ln in body),
           r"\end{table}"]
    tex_path = os.path.join(out_dir, f"{stem}.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(tex) + "\n")

    print(f"\nwrote {csv_path}\n      {tex_path}")
    return rows


if __name__ == "__main__":
    snr_sweep_table("results/snr_sweep/ecHTtests", kind="echt")
