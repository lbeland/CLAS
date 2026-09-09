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

    from analysis.sweep import analyse_snr_sweep, snr_sweep_table
    snr_sweep_table("results/snr_sweep/ecHTtests", kind="echt")
    snr_sweep_table("results/snr_sweep/f0tests",   kind="f0")
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


def _phase_error_stats(values: np.ndarray, times: np.ndarray,
                       trim_frac: float = 0.10) -> "tuple[float, float, int]":
    """Circular mean and circular SD (both in degrees) of a wrapped phase-error
    series, after dropping non-finite samples and trimming trim_frac of the
    record's time span off each end to skip filter edge transients.

    The trim window matches plot_errors(): take the timestamps at the
    trim_frac / (1 - trim_frac) index positions and keep only samples inside
    [t_lo, t_hi]. Circular statistics (scipy.stats.circmean / circstd) are used
    because the error is an angle: the mean stays a meaningful lead/lag bias
    even for a broad distribution, and the SD isn't inflated by the (-180, 180]
    wrap. With this trim and these statistics the numbers match the polar-plot
    annotations in plot_errors()."""
    vals   = np.asarray(values, dtype=float)
    times  = np.asarray(times, dtype=float)
    finite = np.isfinite(vals)
    if times.size:
        lo_idx = int(trim_frac * len(times))
        hi_idx = min(int((1.0 - trim_frac) * len(times)), len(times) - 1)
        t_lo, t_hi = times[lo_idx], times[hi_idx]
        keep = finite & (times >= t_lo) & (times <= t_hi)
    else:
        keep = finite
    vals = vals[keep]
    if vals.size == 0:
        return np.nan, np.nan, 0
    phi  = np.radians(vals)
    mean = float(np.degrees(circmean(phi, high=np.pi, low=-np.pi)))
    std  = float(np.degrees(circstd(phi,  high=np.pi, low=-np.pi)))
    return mean, std, vals.size


def _run_carrier_f0(opts: dict, fallback: float) -> float:
    """The known carrier frequency of a synthetic run
    (SimulatedSource.options.carrier_frequency), or `fallback` if unset."""
    try:
        return float(opts["carrier_frequency"])
    except (KeyError, TypeError, ValueError):
        return fallback

def _hz_error_stats(values: np.ndarray, times: np.ndarray,
                    warmup_s: float = 10.0) -> "tuple[float, float, float, float, float]":
    """(signed mean, mean |err|, SD of |err|, RMSE, valid_ratio) in Hz for an
    f0-error series, over the record *after* its first warmup_s seconds.

    The f0 estimator produces nothing until its sliding buffer has filled
    (~warmup_s), so that opening stretch isn't counted either way: it's
    excluded from the stats and it's not in the valid_ratio denominator.
    valid_ratio is finite estimates / all estimates in the post-warm-up
    window -- 1.0 means the estimator returned a value at every sample once
    it was running, lower means it flagged some samples invalid."""
    vals  = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.size:
        vals = vals[times - times[0] >= warmup_s]
    total = vals.size
    vals  = vals[np.isfinite(vals)]
    if total == 0 or vals.size == 0:
        return np.nan, np.nan, np.nan, np.nan, (np.nan if total == 0 else 0.0)
    return (float(np.mean(vals)), float(np.mean(np.abs(vals))),
            float(np.std(np.abs(vals))), float(np.sqrt(np.mean(vals ** 2))),
            vals.size / total)


# Per-kind table layout: (spec, plain-text header, LaTeX header, per-value format).
# `spec` is a row key, or a (mean_key, std_key) pair rendered as "mean +/- std"
# ("$mean \\pm std$" in LaTeX).
_SWEEP_TABLE_COLUMNS = {
    "echt": [
        ("noise",                                 "noise",               "Noise",                              "{}"),
        ("snr_db",                                "SNR (dB)",             "SNR [dB]",                           "{:g}"),
        (("err_mean", "err_std"),                 "phase err (deg)",      r"$\hat\theta - \theta$",             "{:.2f}"),
        (("hilb_vs_gt_mean", "hilb_vs_gt_std"),   "Hilbert-vs-GT (deg)",  r"$\theta_{\text{HT}} - \theta$",     "{:.2f}"),
        (("vs_hilbert", "vs_hilbert_std"),        "vs-Hilbert (deg)",     r"$\hat\theta - \theta_{\text{HT}}$", "{:.2f}"),
    ],
    "f0": [
        ("noise",       "noise",              "Noise",                             "{}"),
        ("snr_db",      "SNR (dB)",           "SNR [dB]",                          "{:g}"),
        ("signed_mean", "mean err (Hz)",      "mean err (Hz)",                     "{:.3f}"),
        ("abs_mean",    "abs err mean (Hz)",  r"$\overline{|\Delta f_0|}$ (Hz)",   "{:.3f}"),
        ("abs_std",     "SD (Hz)",            "SD (Hz)",                           "{:.3f}"),
        ("rmse",        "RMSE (Hz)",          "RMSE (Hz)",                         "{:.3f}"),
        ("valid_ratio", "valid ratio",        "valid ratio",                       "{:.3f}"),
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
             r"mean indicates a systematic phase lead ($>0$) or lag ($<0$); the "
             r"first and last 10\% of each run are discarded as filter edge "
             r"transients."),
    "f0":   (r"Offline $f_0$-estimation error across the SNR / noise-colour "
             r"sweep: signed mean error, mean absolute error $\pm$ SD, RMSE, and "
             r"the fraction of valid estimates after the estimator's warm-up."),
}


def snr_sweep_table(sweep_dir: str, kind: str = "echt", f0: float = 10,
                    trim_frac: float = 0.10, out_dir: str = None,
                    f0_is_truth: bool = True, whiten: bool = True,
                    caption: str = None, label: str = None) -> "list[dict]":
    """Summarise an SNR / noise-colour sweep as a table instead of a plot.

    One row per run folder under sweep_dir, keyed by that run's SimulatedSource
    snr_db / noise_color:

      kind="echt": circular mean phase error +/- circular SD (deg) from the
                   "Phase error" series (online ecHT vs. ground-truth phase),
                   plus the same for the online-vs-Hilbert error. Errors carry
                   the (estimate - reference) sign, i.e. theta_hat - theta, and
                   match plot_errors()'s polar annotations.
      kind="f0":   signed mean, mean |error| +/- SD and RMSE (Hz) from the
                   "f0 error" series (FrequencyEstimation vs. true inst. freq),
                   plus valid_ratio = finite estimates / all estimates once the
                   estimator is running (its first window_size_sec seconds,
                   read from the run's graph, are excluded as buffer warm-up).

    f0_is_truth (default True): take each run's known carrier_frequency as the
    pre-Hilbert bandpass centre instead of re-estimating f0 offline (see
    analysis.main.load_and_analyse). Pass False to fall back to offline
    estimation with `f0` as the default. whiten (default True) keeps the 1/f
    aperiodic whitening of the offline reference on regardless; pass
    whiten=False to compare against the un-whitened reference.

    trim_frac drops that fraction of the record's time span off each end of the
    phase-error series to skip filter edge transients (kind="echt" only; same
    window as plot_errors()); the f0 error is not trimmed.

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

            if kind == "echt":
                series = next((e for e in errors if e["label"] == "Phase error"), None)
                if series is None:
                    print("  No 'Phase error' series; skipping.")
                    continue
                e_mean, e_std, n = _phase_error_stats(series["values"], series["time_s"], trim_frac)
                if n == 0:
                    print("  No finite phase-error samples; skipping.")
                    continue
                cmp = next((e for e in errors if e["label"] == "Online vs Hilbert"), None)
                vh_mean, vh_std, _ = (_phase_error_stats(cmp["values"], cmp["time_s"], trim_frac)
                                      if cmp is not None else (np.nan, np.nan, 0))
                hgt = next((e for e in errors if e["label"] == "Hilbert ref error"), None)
                hgt_mean, hgt_std, _ = (_phase_error_stats(hgt["values"], hgt["time_s"], trim_frac)
                                        if hgt is not None else (np.nan, np.nan, 0))
                # compute_errors() already uses the (estimate - reference) sign,
                # i.e. these means are circmean(theta_hat - theta) directly.
                rows.append({"noise": noise_type, "snr_db": snr_db,
                             "err_mean": e_mean, "err_std": e_std,
                             "vs_hilbert": vh_mean, "vs_hilbert_std": vh_std,
                             "hilb_vs_gt_mean": hgt_mean, "hilb_vs_gt_std": hgt_std})
                print(f"  {noise_type}, SNR {snr_db:g} dB: phase err "
                      f"{e_mean:.2f} +/- {e_std:.2f} deg; "
                      f"online-vs-Hilbert {vh_mean:.2f} +/- {vh_std:.2f} deg; "
                      f"Hilbert-vs-GT {hgt_mean:.2f} +/- {hgt_std:.2f} deg")
            else:  # "f0"
                series = next((e for e in errors if e["label"] == "f0 error"), None)
                if series is None:
                    print("  No 'f0 error' series; skipping.")
                    continue
                warmup_s = float(procs.get("FrequencyEstimation", {})
                                 .get("options", {}).get("window_size_sec", 10.0))
                s_mean, a_mean, a_std, rmse, valid_ratio = _hz_error_stats(
                    series["values"], series["time_s"], warmup_s)
                if not np.isfinite(a_mean):
                    print("  No valid f0-error samples after warm-up; skipping.")
                    continue
                rows.append({"noise": noise_type, "snr_db": snr_db, "signed_mean": s_mean,
                             "abs_mean": a_mean, "abs_std": a_std, "rmse": rmse,
                             "valid_ratio": valid_ratio})
                print(f"  {noise_type}, SNR {snr_db:g} dB: f0 err {s_mean:.3f} Hz, "
                      f"|f0 err| {a_mean:.3f} +/- {a_std:.3f} Hz, RMSE {rmse:.3f} Hz "
                      f"(valid {valid_ratio:.1%} after {warmup_s:g}s warm-up)")
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
    snr_sweep_table("results/snr_sweep/f0tests",   kind="f0")
