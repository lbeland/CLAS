"""
Plotting and EDF export.
"""

import csv
import datetime
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd, linregress
from scipy.signal import spectrogram, butter, sosfiltfilt, welch, periodogram, windows
import mne
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# Saving to a ".pgf" filename invokes the pgf backend automatically, so the
# default (interactive) backend stays active and plt.show() keeps working.
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

from .stimulus import get_edges

PLOTS_DIR = "/home/linda/Documents/MA/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

TEXTWIDTH    = 6.30045
ASPECT_RATIO = 9/16
SCALE        = 1.0
FIG_WIDTH    = TEXTWIDTH * SCALE
FIG_HEIGHT   = FIG_WIDTH * ASPECT_RATIO
FIGSIZE      = (FIG_WIDTH, FIG_HEIGHT)


def save_pgf(fig, name: str) -> None:
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.pgf"), bbox_inches="tight")

def save_pdf(fig, name: str) -> None:
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.pdf"), bbox_inches="tight", dpi=500)

def save_png(fig, name: str) -> None:
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.png"), bbox_inches="tight", dpi=300)


COMMON_BBOX = dict(
    facecolor="white",
    edgecolor="0.8",
    boxstyle="round,pad=0.2",
    alpha=0.85,
)

CHANNEL_NAMES = {
    1: "Fp1",
    2: "Fpz",
    3: "Fz",
    4: "F3",
    5: "F7",
    27: "F4",
    29: "Fp2",
    28: "F8",
    9: "C3",
    25: "C4",
    6: "E1",
    30: "E2",
    10: "Cz",
    26: "FCz",
    12: "M1",
    23: "M2",
}

# E1/E2 are EOG (eye) channels, not scalp positions -- excluded from CSD/
# surface-Laplacian computations even though they're in CHANNEL_NAMES.
EOG_CHANNEL_NAMES = ("E1", "E2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_polar_axis(ax, r_max, radial_ticks):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=[""] * 8)
    ax.set_ylim(0, r_max)
    ax.set_yticks(radial_ticks)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_yticklabels([rf"{t:.0f}\%" for t in radial_ticks])
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=1.0)
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_rlabel_position(90)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("center")
    r0 = r_max * 1.15
    ax.text(np.deg2rad(45),  r0, r"$+45^\circ$", ha="center", va="center")
    ax.text(np.deg2rad(-45), r0, r"$-45^\circ$", ha="center", va="center")


def _circ_stats(phi_rad):
    phi = np.asarray(phi_rad, float)
    if phi.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mu  = float(circmean(phi, high=np.pi, low=-np.pi))
    sd  = float(circstd(phi,  high=np.pi, low=-np.pi))
    plv = float(np.abs(np.mean(np.exp(1j * phi))))
    pli = float(np.abs(np.mean(np.sign(np.sin(phi)))))
    return mu, sd, plv, pli


def _edge_trim_window(errors: list[dict], lo_frac: float = 0.1,
                      hi_frac: float = 0.90) -> tuple[float, float]:
    """The [t_lo, t_hi] wall-clock window that drops the leading/trailing
    edge-transient slice of a recording.

    Derived from the *timestamps* of the reference series (errors[0]) at
    lo_frac / hi_frac of its length, so every series -- regardless of its own
    sampling rate -- is trimmed by the same amount of *time*, and sparse
    series (e.g. one value per stimulus) aren't sliced away to nothing.

    Both plot_errors() (which trims before computing polar statistics) and the
    "phase_error" stack panel (which draws t_lo/t_hi as dashed markers) call this,
    so the trim shown always matches the trim applied. Returns (-inf, +inf)
    when there is nothing to trim on.
    """
    if not errors:
        return (-np.inf, np.inf)
    ref_t = np.asarray(errors[0]["time_s"], dtype=float)
    if ref_t.size == 0:
        return (-np.inf, np.inf)
    lo_idx = int(lo_frac * len(ref_t))
    hi_idx = min(int(hi_frac * len(ref_t)), len(ref_t) - 1)
    return float(ref_t[lo_idx]), float(ref_t[hi_idx])


def _write_error_table(rows: "list[tuple[str, float, float, float]]", name: str = "error_distributions",
                       caption: str = "Circular mean $\\pm$ SD and PLV for different phase error series.",
                       label: str = "tab:error_distributions") -> None:
    """Write a three-column LaTeX table (series label, mean +/- SD in degrees,
    PLV) of the circular phase-error statistics to PLOTS_DIR/<name>.tex.

    rows are (label, mean_deg, sd_deg, plv) tuples; mean/sd are the same
    numbers plot_errors() annotates on each polar panel, so the table and the
    figure always agree. PLV = |E[e^{j PE}]|, PE the (wrapped, radian) phase
    error -- see _circ_stats().
    """
    body = [r"\begin{tabular}{lrr}", r"\toprule",
            r"Phase Error & Mean $\pm$ SD [$^\circ$] & PLV \\", r"\midrule"]
    for lbl, mu, sd, plv in rows:
        cell = rf"${mu:.1f}^\circ \pm {sd:.1f}^\circ$" if np.isfinite(mu) and np.isfinite(sd) else "--"
        plv_cell = f"{plv:.2f}" if np.isfinite(plv) else "--"
        body.append(rf"{lbl} & {cell} & {plv_cell} \\")
    body += [r"\bottomrule", r"\end{tabular}"]

    tex = [r"\begin{table}[htbp]", r"  \centering", rf"  \caption{{{caption}}}",
           rf"  \label{{{label}}}", *("  " + ln for ln in body), r"\end{table}"]
    tex_path = os.path.join(PLOTS_DIR, f"{name}.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"wrote {tex_path}")


def write_condition_error_table(
    condition_stats: "dict[int, dict[str, tuple[float, float, float]]]",
    columns: list[str],
    condition_groups: "list[list[int]]",
    condition_labels: "dict[int, str]",
    name: str = "condition_error_table",
    caption: str = r"Circular mean $\pm$ SD and PLV of each error metric, per stimulus-onset condition.",
    label: str = "tab:condition_error_table",
) -> None:
    """Write a LaTeX table -- one row per stim_onset_deg condition, two
    subcolumns per error label (circular mean +/- SD, and PLV) under a
    spanning header -- to PLOTS_DIR/<name>.tex, and print the same table to
    the console.

    condition_stats: {condition: {error_label: (mean_deg, sd_deg, plv)}}, as
    built by analyse_pooled_errors() via _circ_stats() -- every column here is
    a degree-unit circular metric (mean/sd in degrees, plv the resultant
    vector length |E[e^{j*error}]|). PLV gets its own subcolumn (via
    \\multicolumn/\\cmidrule) rather than being folded into the mean+-SD cell,
    since that overflows textwidth once there are more than a couple of error
    labels. condition_labels maps each condition (e.g. stim_onset_deg in
    degrees) to a short display name (e.g. 330 -> "Pre-Peak").
    condition_groups fixes the row order as blocks of conditions -- e.g.
    [[60, 150, 240, 330], [0, 180]] to put the four phase-offset conditions
    first, then a rule, then the two P1 (peak/trough) conditions -- with a
    rule drawn between consecutive groups.
    """
    def mean_sd_cell(condition: int, col: str, tex: bool = False) -> str:
        stat = condition_stats.get(condition, {}).get(col)
        if stat is None or not np.isfinite(stat[0]):
            return "--"
        mu, sd, _ = stat
        return rf"${mu:.1f}^\circ \pm {sd:.1f}^\circ$" if tex else f"{mu:.1f}° ± {sd:.1f}°"

    def plv_cell(condition: int, col: str) -> str:
        stat = condition_stats.get(condition, {}).get(col)
        plv = stat[2] if stat is not None else np.nan
        return f"{plv:.2f}" if np.isfinite(plv) else "--"

    def row_name(condition: int) -> str:
        return f"{condition_labels.get(condition, str(condition))} ({condition}°)"

    condition_order = [c for group in condition_groups for c in group]
    # Row index (into condition_order) right before which a rule is drawn --
    # i.e. the running length of every group but the last.
    rule_before = set()
    n = 0
    for group in condition_groups[:-1]:
        n += len(group)
        rule_before.add(n)

    # Console table, plain text with fixed-width columns. Each error label
    # gets two adjacent columns (Mean +/- SD, PLV); the label itself is
    # printed once, left-aligned over its Mean +/- SD subcolumn, with the PLV
    # subcolumn left blank in that header row.
    header_row1 = ["", *(v for col in columns for v in (col, ""))]
    header_row2 = ["Condition", *(v for _ in columns for v in ("Mean ± SD", "PLV"))]
    data_rows   = [[row_name(c), *(v for col in columns for v in (mean_sd_cell(c, col), plv_cell(c, col)))]
                   for c in condition_order]
    text_rows   = [header_row1, header_row2] + data_rows
    col_widths  = [max(len(r[i]) for r in text_rows) for i in range(len(header_row2))]
    print()
    for i, row in enumerate(text_rows):
        if i == 2:
            print("  ".join("-" * w for w in col_widths))
        elif i > 2 and (i - 2) in rule_before:
            print("  ".join("-" * w for w in col_widths))
        print("  ".join(v.ljust(w) for v, w in zip(row, col_widths)))
    print()

    # LaTeX table: a spanning label row (\multicolumn + \cmidrule) over each
    # label's two subcolumns, then a "Mean +/- SD" / "PLV" subheader row.
    col_spec    = "l" + "rr" * len(columns)
    label_cells = ["", *(rf"\multicolumn{{2}}{{c}}{{{col}}}" for col in columns)]
    cmidrules   = " ".join(rf"\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(len(columns)))
    subheader   = ["Condition", *(v for _ in columns for v in (r"Mean $\pm$ SD", "PLV"))]
    body = [rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
            " & ".join(label_cells) + r" \\", cmidrules,
            " & ".join(subheader) + r" \\", r"\midrule"]
    for i, condition in enumerate(condition_order):
        if i in rule_before:
            body.append(r"\midrule")
        tex_cells = [row_name(condition),
                     *(v for col in columns for v in (mean_sd_cell(condition, col, tex=True), plv_cell(condition, col)))]
        body.append(" & ".join(tex_cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]

    tex = [r"\begin{table}[htbp]", r"  \centering", rf"  \caption{{{caption}}}",
           rf"  \label{{{label}}}", *("  " + ln for ln in body), r"\end{table}"]
    tex_path = os.path.join(PLOTS_DIR, f"{name}.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"wrote {tex_path}")


# ---------------------------------------------------------------------------
# Error plots
# ---------------------------------------------------------------------------

def plot_errors(errors: list[dict]) -> None:
    """Polar error-distribution histograms -- one per 'degrees'-unit series,
    with the circular mean +/- SD annotated.

    No time-window trim is applied here: hilbert_phase/f0_continuous already
    carry NaN on their own leading/trailing edges (see
    analysis.main.load_and_analyse), so any series derived from them already
    excludes those samples by the time it reaches this function, while a
    series that doesn't touch the Hilbert reference (e.g. an online estimate
    scored against true_phase) is used in full. Dropping NaNs below is enough
    either way -- single-recording and pooled (analyse_pooled_errors())
    `errors` lists are handled identically.
    """
    if not errors:
        print("No errors to plot.")
        return

    deg_indices  = [i for i, err in enumerate(errors) if err["unit"] == "degrees"]
    n_polar_rows = 2 if len(deg_indices) > 4 else 1
    n_polar_cols = int(np.ceil(len(deg_indices) / n_polar_rows)) if deg_indices else 1
    fig_polars = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT * n_polar_rows))
    ax_polars  = {i: fig_polars.add_subplot(n_polar_rows, n_polar_cols, j + 1, projection="polar")
                  for j, i in enumerate(deg_indices)}

    colors        = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bin_width_deg = 10
    bin_width_rad = np.radians(bin_width_deg)
    bins          = np.arange(-180, 181, bin_width_deg)
    centers       = (np.radians(bins[:-1]) + np.radians(bins[1:])) / 2

    hists       = {}
    trimmed_rad = {}
    for i in deg_indices:
        vals = np.asarray(errors[i]["values"], dtype=float)
        vals = vals[~np.isnan(vals)]
        hists[i]       = np.histogram(vals, bins=bins)[0] / max(1, vals.size) * 100
        trimmed_rad[i] = np.radians(vals)

    r_max   = max(h.max() for h in hists.values()) * 1.05 if hists else 5
    r_max   = max(r_max, 5)
    # "nice" radial ticks that stay evenly spread whatever r_max is, instead of
    # a fixed list that bunches up near the centre for large r_max.
    r_ticks = [t for t in MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(0, r_max)
               if 0 < t < r_max]

    table_rows = []
    for i, err in enumerate(errors):
        if err["unit"] != "degrees":
            continue
        if np.nansum(np.abs(err["values"])) == 0:
            continue
        label_lower = err["label"].lower()
        if "onset" in label_lower:
            color = "orange"
        elif "offset" in label_lower:
            color = "green"
        else:
            color = colors[i % len(colors)]
        mu_u, sd_u, plv, _ = _circ_stats(trimmed_rad[i])
        mu_deg, sd_deg = np.degrees(mu_u) + 0.0, np.degrees(sd_u)
        table_rows.append((err["label"], mu_deg, sd_deg, plv))

        ax_polars[i].bar(centers, hists[i], width=bin_width_rad,
                         color=color, edgecolor="0", linewidth=0.75)
        ax_polars[i].plot([mu_u, mu_u], [0, r_max], color="0", linewidth=2)
        _style_polar_axis(ax_polars[i], r_max, r_ticks)
        ax_polars[i].text(
            0.5, 0.45,
            rf"${np.round(mu_deg, 1):.1f}^\circ"
            rf" \pm {np.round(sd_deg, 1):.1f}^\circ$",
            transform=ax_polars[i].transAxes, bbox=COMMON_BBOX, va="top", ha="center",
        )
        ax_polars[i].set_title(err["label"])

    fig_polars.tight_layout()
    save_pgf(fig_polars, "error_distributions")
    save_pdf(fig_polars, "error_distributions")
    _write_error_table(table_rows)


def plot_pooled_scatter(records: list[dict]) -> None:
    """Per-recording scatter for pooled-error analysis: one bullet per
    recording at the circular mean of its online-vs-Hilbert phase error
    (y, with the circular SD as a vertical error bar), against a scalar
    x for that recording --

      1. the variance of its online (Kalman) f0 estimate  [Hz^2]
      2. its offline alpha-peak SNR                        [dB]

    Each record dict carries the raw online-vs-Hilbert series ("phase_vals"),
    plus "f0_var" and "snr_db". The y statistics drop non-finite samples
    below (_circ_stats() itself assumes a NaN-free series); the series' edge
    transients are already NaN by then since it's Hilbert-derived (see
    analysis.main.load_and_analyse).
    """
    if not records:
        print("No per-recording scatter data to plot.")
        return

    stats = []
    for r in records:
        vals = np.asarray(r["phase_vals"], dtype=float)
        vals = vals[np.isfinite(vals)]
        mu_u, sd_u, _, _ = _circ_stats(np.radians(vals))
        stats.append((np.degrees(mu_u), np.degrees(sd_u)))
    stats   = np.array(stats, dtype=float)
    mu, sd  = stats[:, 0], stats[:, 1]
    f0_var  = np.array([r.get("f0_var", np.nan) for r in records], dtype=float)
    snr_db  = np.array([r.get("snr_db", np.nan) for r in records], dtype=float)

    ylabel = r"$\hat\theta - \theta_{\mathrm{HT}}$ [$^\circ$]"
    for x, xlabel, name in (
        (f0_var, r"$\hat f_0$ variance [Hz$^2$]", "pooled_phaseerr_vs_f0var"),
        (snr_db, "SNR [dB]",                     "pooled_phaseerr_vs_snr"),
    ):
        ok = np.isfinite(mu) & np.isfinite(x)
        if not ok.any():
            print(f"plot_pooled_scatter: no finite points for {xlabel!r}; skipping.")
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.errorbar(x[ok], mu[ok], yerr=sd[ok], fmt="o", markersize=4,
                    capsize=2, elinewidth=0.8, color="tab:blue", ecolor="0.6")
        ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")

        # Ordinary least-squares fit of the per-recording mean error on x,
        # annotated with the coefficient of determination R^2 (= r**2).
        if np.count_nonzero(ok) >= 2:
            fit = linregress(x[ok], mu[ok])
            xr  = np.array([x[ok].min(), x[ok].max()])
            ax.plot(xr, fit.intercept + fit.slope * xr, color="tab:red", linewidth=1.2,
                    label=rf"OLS fit, $R^2 = {fit.rvalue ** 2:.2f}$")
            ax.legend(loc="upper right")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)
        fig.tight_layout()
        save_pdf(fig, name)
        save_pgf(fig, name)


# ---------------------------------------------------------------------------
# Stackable time-domain panels
# ---------------------------------------------------------------------------
#
# Each _panel_* function draws one self-contained time-domain view onto a
# supplied Axes, using data pulled from a RecordingAnalysis (see
# analysis/main.py). They all put "seconds since start_ts" on the x-axis, so
# any subset can be stacked on a shared time axis by plot_stack().
#
# Panel names (as passed to plot_stack):
#   "f0"          - f0 over time: true / offline-Hilbert / MODAL / raw / smoothed
#   "phase"       - offline Hilbert / online / true phase estimates
#   "phase_error" - angular (degree) error series: phase & stimulus-edge errors
#   "signals"     - raw & online-filtered EEG with stimulus/trigger shading


def _panel_f0(ax, a, start_ts: float, time_range: tuple = None) -> None:
    """f0 over time: true, offline-Hilbert, MODAL, raw peak estimate, smoothed estimate."""
    gt, samples = a.ground_truth, a.samples
    time_s      = (gt["time"] - start_ts) / 1e6

    if gt.get("true_inst_freq") is not None:
        y = gt["true_inst_freq"]
        m = min(len(time_s), len(y))
        ax.plot(time_s[:m], y[:m], linewidth=1.5, alpha=0.8, label=r"$f_0$", rasterized=True)

    if a.f0_continuous is not None:
        y = a.f0_continuous
        m = min(len(time_s), len(y))
        ax.plot(time_s[:m], y[:m], linewidth=1.2, alpha=0.7,
                label=r"$\hat f_0$ from $\theta_{\mathrm{HT}}$", rasterized=True)

    if a.f0_modal is not None:
        y = a.f0_modal
        m = min(len(time_s), len(y))
        ax.plot(time_s[:m], y[:m], linewidth=1.2, alpha=0.7, linestyle="--",
                label=r"$\hat f_0$ MODAL", rasterized=True)

    if samples.get("FrequencyEstimation_raw") is not None:
        x_raw = (samples["FrequencyEstimation_raw"]["x"] - start_ts) / 1e6
        ax.plot(x_raw, samples["FrequencyEstimation_raw"]["y"], "o", markersize=1.5,
                color="tab:red", alpha=0.5, label=r"Raw $\hat f_0$", rasterized=True)

    if samples.get("FrequencyEstimation") is not None:
        x_est = (samples["FrequencyEstimation"]["x"] - start_ts) / 1e6
        ax.plot(x_est, samples["FrequencyEstimation"]["y"], linewidth=1.3,
                label=r"Kalman $\hat f_0$", rasterized=True)

    ax.set_ylabel("Frequency [Hz]")
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.minorticks_on()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)
    ax.set_ylim(2, 20)
    ax.legend(loc="lower right")


def _panel_phase_error(ax, a, start_ts: float, time_range: tuple = None) -> None:
    """Angular (degree) error time series: phase errors and stimulus-edge errors.
    The dashed markers come from the shared _edge_trim_window() and mark where
    hilbert_phase/f0_continuous become NaN at their own edges (see
    analysis.main.load_and_analyse) -- i.e. where Hilbert-derived series in
    this panel start/stop being defined, not a trim applied to every series."""
    if not any(err["unit"] == "degrees" for err in a.errors):
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    t_lo, t_hi = _edge_trim_window(a.errors)
    if np.isfinite(t_lo):
        ax.axvline(x=t_lo, color="0.5", linewidth=0.8, linestyle="--")
        ax.axvline(x=t_hi, color="0.5", linewidth=0.8, linestyle="--")

    for i, err in enumerate(a.errors):
        if err["unit"] != "degrees":
            continue
        ax.plot(err["time_s"], err["values"], linestyle=err.get("linestyle", "-"),
                linewidth=1, color=colors[i % len(colors)],
                label=err["label"], alpha=0.85, rasterized=True)

    ax.legend(frameon=True, loc="upper right")
    ax.set_ylabel("Error [°]")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)


def _panel_signals(ax, a, start_ts: float, time_range: tuple = None) -> None:
    """Raw & online-filtered EEG with target-stim / trigger / stimulus shading."""
    gt, samples = a.ground_truth, a.samples
    time_s = (gt["time"] - start_ts) / 1e6
    n      = len(time_s)

    def _pick(label):
        d = samples.get(label)
        return d["y"] if d else None

    raw_eeg     = gt["raw"] - np.mean(gt["raw"])
    filt_online = _pick("ecHTFilter")
    stimulus    = _pick("StimControl")
    trigger     = _pick("UDPSource_TRIGGER")

    ax.plot(time_s, raw_eeg[:n], color="0.6", label="Raw", rasterized=True)
    if filt_online is not None:
        ax.plot(time_s, filt_online[:n], color="0.3", label="Filtered (online)", rasterized=True)
    if a.stim_ref is not None:
        ax.fill_between(time_s, 0, 1, where=a.stim_ref[:n], color="tab:blue", alpha=0.4,
                        transform=ax.get_xaxis_transform(), label="Target stim")
    if trigger is not None:
        ax.fill_between(time_s, 0, 1, where=trigger[:n], color="tab:orange", alpha=0.4,
                        transform=ax.get_xaxis_transform(), label="Trigger Stim")
    elif stimulus is not None:
        ax.fill_between(time_s, 0, 1, where=stimulus[:n], color="tab:orange", alpha=0.4,
                        transform=ax.get_xaxis_transform(), label="Stimulus")

    ax.set_ylabel("Amplitude")
    ax.legend(facecolor="white", frameon=True, loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)


def _panel_phase(ax, a, start_ts: float, time_range: tuple = None) -> None:
    """Offline Hilbert / online / true phase estimates."""
    gt, samples   = a.ground_truth, a.samples
    time_s        = (gt["time"] - start_ts) / 1e6
    n             = len(time_s)
    hilbert_phase = a.hilbert_phase
    online_phase  = samples.get("PhaseEstimation_phase")
    online_phase  = online_phase["y"] if online_phase else None
    true_phase    = gt.get("true_phase")

    if hilbert_phase is not None:
        ax.plot(time_s[:len(hilbert_phase)], hilbert_phase[:n], color="tab:blue",
                linewidth=1.5, label="Offline Hilbert phase", rasterized=True)
    if online_phase is not None:
        ax.plot(time_s[:len(online_phase)], online_phase[:n], color="tab:orange",
                linewidth=1.5, label="Online phase", rasterized=True)
    if true_phase is not None:
        ax.plot(time_s[:len(true_phase)], true_phase[:n], color="tab:brown",
                linewidth=1.0, label="True phase", rasterized=True)

    ax.set_ylabel("Phase (rad)")
    ax.legend(facecolor="white", frameon=True, loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)


# name -> drawer
_PANEL_DRAWERS = {
    "f0":          _panel_f0,
    "phase":       _panel_phase,
    "phase_error": _panel_phase_error,
    "signals":     _panel_signals,
}


def plot_stack(analysis, panels: list[str], *, start_ts: float = None,
               time_range: tuple = None, save_as: str = None,
               height_ratios: list[float] = None):
    """Stack any subset of the per-recording time-domain panels in one figure,
    sharing a common time axis. No titles are drawn.

    analysis:      a RecordingAnalysis (see analysis/main.py).
    panels:        ordered panel names, top to bottom. Valid names:
                   "f0", "phase", "phase_error", "signals".
    start_ts:      x-axis origin in ground_truth["time"] units; defaults to the
                   recording start (ground_truth["time"][0]).
    height_ratios: per-panel vertical weights; defaults to an even split.
    save_as:       basename for PDF an under PLOTS_DIR (skipped when None).

    Returns (fig, axes).
    """
    unknown = [p for p in panels if p not in _PANEL_DRAWERS]
    if unknown:
        raise ValueError(f"Unknown panel(s) {unknown}; valid: {list(_PANEL_DRAWERS)}")
    if not panels:
        raise ValueError("plot_stack() needs at least one panel.")

    if start_ts is None:
        start_ts = analysis.ground_truth["time"][0]
    if height_ratios is None:
        height_ratios = [1.0] * len(panels)

    fig, axes = plt.subplots(
        len(panels), 1, sharex=True, squeeze=False,
        figsize=(FIG_WIDTH, FIG_HEIGHT * 0.7 * sum(height_ratios)),
        height_ratios=height_ratios,
    )
    axes = axes[:, 0]

    for i, (ax, name) in enumerate(zip(axes, panels)):
        _PANEL_DRAWERS[name](ax, analysis, start_ts, time_range)
        if len(panels) > 1:
            ax.set_title(f"({chr(ord('a') + i)})")

    axes[-1].set_xlabel("Time [s]")
    if time_range is not None:
        axes[-1].set_xlim(time_range)
        # axes[-1].set_ylim(-2.5,2.5)
    fig.tight_layout()
    if save_as:
        save_pdf(fig, save_as)
    return fig, axes

# ---------------------------------------------------------------------------
# Spectrum and time-series plots
# ---------------------------------------------------------------------------

def plot_spectrum(raw: np.ndarray, samples: dict, filtered: np.ndarray, fs: float, aperiodic_params: tuple = None, X_white: np.ndarray = None) -> None:
    """Plot FFT magnitude spectrum of raw and filtered signals."""
    freqs = np.fft.rfftfreq(len(raw), d=1 / fs)

    # aperiodic_params were fit to a density-scaled PSD (Welch), where
    # Pxx(f) = 2*|X(f)|^2 / (fs*n). Rescale the intercept to raw FFT power
    # units: L(f) ~ |X(f)|^2. Take sqrt(L) to compare against the amplitude
    # spectrum plotted below (same convention as compute_hilbert_reference).
    # aperiodic_params is None when f0 was taken as ground truth (no 1/f fit).
    L = None
    if aperiodic_params is not None:
        slope, intercept = aperiodic_params
        n = len(raw)
        intercept_fft = intercept + np.log10(fs * n / 2)
        L = freqs[1:] ** slope * 10 ** intercept_fft  # skip f=0

    # plt.figure()
    # plt.plot(np.log10(freqs[1:]), np.log10(np.abs(np.fft.rfft(raw))[1:]), label="Raw", alpha=0.7, color="blue")
    # plt.plot(np.log10(freqs[1:]), np.log10(np.sqrt(L)), label="Aperiodic fit", alpha=0.7, color="red", linestyle="--")


    nrows = 2 if X_white is not None else 1
    fig, axes = plt.subplots(nrows, 1, figsize=FIGSIZE, squeeze=False, sharex=True)
    ax = axes[0, 0]
    taper = windows.tukey(len(raw), alpha=0.01)
    X = np.abs(np.fft.rfft(raw*taper))
    ax.semilogy(freqs, X, label="Raw", alpha=0.7, color="blue")

    if samples.get("ecHTFilter") is not None:
        filt        = samples["ecHTFilter"]["y"]
        # taper = windows.tukey(len(filt), alpha=0.01)
        # filt_wind = filt *taper
        freqs_filt  = np.fft.rfftfreq(len(filt), d=1 / fs)
        ax.semilogy(freqs_filt, np.abs(np.fft.rfft(filt)), label="Filtered (online)", alpha=0.7, color="orange")

    if L is not None:
        ax.semilogy(freqs[1:], np.sqrt(L), label="Aperiodic fit", alpha=0.7, color="red", linestyle="--")

    if X_white is not None:
        ax2 = axes[1, 0]
        ax2.set_ylabel("Magnitude")
        ax2.semilogy(freqs, np.abs(X_white), label="Whitened", alpha=0.7, color="purple")
        ax2.semilogy(freqs, np.abs(np.fft.rfft(filtered)), label="Filtered (offline)", alpha=0.7, color="green")
        ax2.set_ylim(0, np.max(np.abs(X_white)[(freqs >= 5) & (freqs <= 10)]) * 5)
        ax2.legend(loc="upper right")
    else:
        ax.semilogy(freqs, np.abs(np.fft.rfft(filtered)), label="Filtered (offline)", alpha=0.7, color="green")

    ax.set_xlim(0, 100)
    # Use the maximum between 5-10Hz as xlim
    ax.set_ylim(0, np.max(X[(freqs >= 5) & (freqs <= 10)]) * 5)
    ax.set_ylabel("Magnitude")
    ax.set_title("Spectrum")
    (ax2 if X_white is not None else ax).set_xlabel("Frequency (Hz)")
    # get legend handler from both axis and combine them
    ax.legend(loc="upper right")

    # fig = plt.figure(figsize=FIGSIZE)
    # f, t, Sxx = spectrogram(raw, fs=fs, nperseg=int(fs*30))
    # plt.pcolormesh(t, f, np.log(Sxx), shading='nearest')
    # plt.colorbar(label='Intensity (log scale)')
    # plt.ylim(0, 50)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')

    # save_pgf(fig, "spectrum")

# ---------------------------------------------------------------------------
# EDF export
# ---------------------------------------------------------------------------

def load_stim_annotations(results_dir: str, start_ts: float) -> "mne.Annotations | None":
    """Load stim event markers from a stim protocol log CSV in results_dir, if present.

    Each "on" or "off" row opens a segment that runs until the next logged row
    (whichever state that is), producing a "stim_on"/"stim_off" duration
    annotation. Non on/off rows (protocol_start/end/stopped) additionally get
    their own zero-duration marker.

    start_ts is the recording start time in the same units as ground_truth["time"]
    (microseconds since epoch), used to align log timestamps to the EDF timeline.
    """
    log_files = sorted(glob.glob(os.path.join(results_dir, "stim_protocol_log_*.csv")))
    if not log_files:
        return None

    with open(log_files[-1], newline="") as f:
        reader = csv.DictReader(f)
        rows = [(datetime.datetime.fromisoformat(row["timestamp"]).timestamp() - start_ts / 1e6,
                 row["state"]) for row in reader]

    onsets, durations, descriptions = [], [], []
    prev_time, prev_state = None, None
    for t, state in rows:
        if prev_state in ("on", "off"):
            onsets.append(prev_time)
            durations.append(t - prev_time)
            descriptions.append("stim_on" if prev_state == "on" else "stim_off")
        if state not in ("on", "off"):
            onsets.append(t)
            durations.append(0.0)
            descriptions.append(state)
        prev_time, prev_state = t, state

    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


# EDF/HDF5 export lives in analysis/edf_io.py (write_raw_signals_edf/load_runtime/
# write_analysis_edf) and analysis/runtime_meta.py (write_runtime_metadata/load_runtime_metadata).


# ---------------------------------------------------------------------------
# ERP
# ---------------------------------------------------------------------------

def apply_csd_transform(fs: float, samples: dict, channel: list[int]) -> dict[int, np.ndarray]:
    """Jointly re-reference every requested channel with a known scalp
    position (see CHANNEL_NAMES) using the CSD (surface Laplacian)
    transform, returning {channel: csd_y}.

    Channels without a known position, and EOG channels (E1/E2), are
    omitted -- callers should fall back to the as-recorded signal for those.
    """
    scalp_channels = [ch for ch in channel
                       if ch in CHANNEL_NAMES and CHANNEL_NAMES[ch] not in EOG_CHANNEL_NAMES]
    if not scalp_channels:
        return {}

    ys = []
    for ch in scalp_channels:
        ch0   = ch - 1
        entry = samples.get(f"UDPSource_{ch0}") or samples.get(f"SimulatedSource_{ch0}")
        ys.append(entry["y"])

    min_len = min(len(y) for y in ys)
    data    = np.array([y[:min_len] for y in ys]) * 1e-6  # uV -> V, mne's expected EEG unit

    info = mne.create_info([CHANNEL_NAMES[ch] for ch in scalp_channels], sfreq=fs,
                            ch_types="eeg", verbose=False)
    raw  = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1020")
    raw  = mne.preprocessing.compute_current_source_density(raw, copy=False)

    csd_data = raw.get_data() * 1e6  # back to uV, matching the untransformed samples
    return {ch: csd_data[i] for i, ch in enumerate(scalp_channels)}


def get_erp_windows(fs: float, samples: dict, channel: list[int] = [1],
                     bandpass: tuple[float, float] = (2.0, 30.0),
                     reject_uv: float = 10000.0,
                     average: bool = False) -> list[dict]:
    """Extract EEG epochs around each trigger onset (-250 ms to +500 ms) for each channel.

    The EEG is bandpass-filtered (default 2-30 Hz) before epoching, and any
    epoch whose peak amplitude exceeds ``reject_uv`` (default ±100 µV) is
    discarded. Each returned window carries a 1-based "channel" key so
    callers can group windows by channel.

    If ``average`` is True, each channel's per-trial epochs are collapsed
    into a single mean window (carrying an added "n_epochs" key) before
    returning, instead of one window per trial -- much lighter to keep
    around when only the averaged ERP is needed (e.g. one curve per subject
    pooled across many recordings).
    """
    all_windows    = []

    for ch in channel:
        ch0 = ch - 1  # zero-based

        # if samples.get("ecHTFilter") is not None:
        #     eeg_y, eeg_t = samples["ecHTFilter"]["y"], samples["ecHTFilter"]["x"]
        if samples.get(f"UDPSource_{ch0}") is not None:
            eeg_y, eeg_t = samples[f"UDPSource_{ch0}"]["y"], samples[f"UDPSource_{ch0}"]["x"]
        elif samples.get(f"SimulatedSource_{ch0}") is not None:
            eeg_y, eeg_t = samples[f"SimulatedSource_{ch0}"]["y"], samples[f"SimulatedSource_{ch0}"]["x"]
        else:
            print(f"Error: no EEG signal found for channel {ch}.")
            continue

        if samples.get("UDPSource_TRIGGER") is not None:
            trigger_y = samples["UDPSource_TRIGGER"]["y"]
        elif samples.get("StimControl") is not None:
            trigger_y = samples["StimControl"]["y"]
        else:
            print("Error: no trigger signal found.")
            return []

        time_s         = (eeg_t - eeg_t[0]) / 1e6
        trigger_binary = (trigger_y > 0.5).astype(float)
        onsets         = get_edges(trigger_binary, "rising")
        offsets        = get_edges(trigger_binary, "falling")

        # Bandpass-filter the continuous EEG before epoching, so filter edge
        # artifacts don't contaminate the epoch boundaries.
        sos   = butter(1, bandpass, btype="band", fs=fs, output="sos")
        eeg_y = sosfiltfilt(sos, eeg_y)

        ch_windows = []
        n_rejected = 0
        for idx in onsets:
            start_idx = idx - int(0.25 * fs)
            end_idx   = idx + int(0.5  * fs)
            if start_idx >= 0 and end_idx < len(eeg_y):
                epoch = eeg_y[start_idx:end_idx]
                if np.max(np.abs(epoch)) > reject_uv:
                    n_rejected += 1
                    continue
                later_offsets = offsets[offsets > idx]
                stim_dur_s    = (time_s[later_offsets[0]] - time_s[idx]) if len(later_offsets) else None
                ch_windows.append({"channel":    ch,
                                   "time_s":     time_s[start_idx:end_idx],
                                   "eeg_y":       epoch,
                                   "stim_dur_s":  stim_dur_s})

        if n_rejected:
            print(f"Channel {ch}: rejected {n_rejected} epoch(s) with peak amplitude > ±{reject_uv:.0f} µV.")

        if average and ch_windows:
            stim_durs = [w["stim_dur_s"] for w in ch_windows if w["stim_dur_s"] is not None]
            all_windows.append({"channel":   ch,
                                "time_s":     ch_windows[0]["time_s"],
                                "eeg_y":      np.mean([w["eeg_y"] for w in ch_windows], axis=0),
                                "stim_dur_s": float(np.mean(stim_durs)) if stim_durs else None,
                                "n_epochs":   len(ch_windows)})
        else:
            all_windows.extend(ch_windows)

    return all_windows


def plot_erp_latency(windows: list[dict], fs: float) -> float | None:
    """Plot per-channel averaged ERPs stacked vertically on a shared x-axis.

    Windows are grouped by their "channel" key (windows without one, e.g.
    from older callers, are grouped together). The reported P1 latency is
    the mean of the per-channel P1 latencies.
    """
    if not windows:
        print("No valid trigger windows found.")
        return None

    channel_order = list(dict.fromkeys(w.get("channel") for w in windows))
    time           = windows[0]["time_s"] - windows[0]["time_s"][int(0.25 * fs)]
    post_stim_mask = (time >= 0.02) & (time <= 0.100)
    window_idx     = np.where(post_stim_mask)[0]

    channels = []
    for ch in channel_order:
        ch_windows = [w for w in windows if w.get("channel") == ch]
        avg_eeg    = np.mean([w["eeg_y"] for w in ch_windows], axis=0)
        std_eeg    = np.std( [w["eeg_y"] for w in ch_windows], axis=0)
        p1_idx     = window_idx[np.argmax(avg_eeg[window_idx])]
        p1_latency = time[p1_idx]
        channels.append({"channel": ch, "windows": ch_windows,
                         "avg": avg_eeg, "std": std_eeg, "p1_latency": p1_latency})
        label = f"channel {ch}" if ch is not None else "all channels"
        print(f"Found {len(ch_windows)} valid trigger windows for {label}; "
              f"P1 latency {p1_latency * 1000:.1f} ms")

    p1_latency = float(np.mean([c["p1_latency"] for c in channels]))
    print(f"Mean P1 latency across {len(channels)} channel(s): {p1_latency * 1000:.1f} ms")

    stim_durs = [w["stim_dur_s"] for w in windows if w["stim_dur_s"] is not None]
    stim_dur  = np.mean(stim_durs) if stim_durs else None
    if stim_dur is not None:
        print(f"Estimated stimulus duration: {stim_dur * 1000:.1f} ms")

    # Vertical spacing between channels, large enough that ±1 SD bands don't overlap.
    # span        = max(np.max(c["avg"] + c["std"]) - np.min(c["avg"] - c["std"]) for c in channels)
    span        = max(np.max(c["avg"]) - np.min(c["avg"]) for c in channels)
    offset_step = span * 1.2

    fig = plt.figure(figsize=FIGSIZE)
    if stim_dur is not None:
        plt.axvspan(0, stim_dur, color="0.6", alpha=0.2, zorder=0, label="Stimulus")

    yticks, yticklabels = [], []
    for i, c in enumerate(channels):
        offset = -i * offset_step
        # for w in c["windows"][:20]:
        #     plt.plot(time, w["eeg_y"] + offset, color="0.8", linewidth=0.8, alpha=0.8)
        plt.plot(time, c["avg"] + offset, color="tab:blue", linewidth=2,
                 label="Average ERP" if i == 0 else None)
        plt.fill_between(time, c["avg"] - c["std"] + offset, c["avg"] + c["std"] + offset,
                         color="tab:blue", alpha=0.2, label="±1 SD" if i == 0 else None)
        yticks.append(offset)
        yticklabels.append(f"{CHANNEL_NAMES.get(c['channel'], c['channel'])}" if c["channel"] is not None else "")

    plt.axvline(0, color="0.0", linestyle="--", label="Trigger onset")
    plt.axvline(p1_latency, color="tab:red", linestyle="--",
               label=f"Mean P1 latency ({p1_latency * 1000:.1f} ms)")

    plt.xlabel("Peri-stimulus time (s)")
    plt.ylabel("Channel" if len(channels) > 1 else "EEG amplitude (µV)")
    if len(channels) > 1:
        plt.yticks(yticks, yticklabels)
    plt.legend(loc="upper right")

    save_pgf(fig, "erp_latency")

    # Second figure: grand average ERP pooling every window from every channel.
    avg_pooled = np.mean([w["eeg_y"] for w in windows], axis=0)
    std_pooled = np.std( [w["eeg_y"] for w in windows], axis=0)
    p1_idx_pooled = window_idx[np.argmax(avg_pooled[window_idx])]
    p1_latency_pooled = time[p1_idx_pooled]
    print(f"Pooled P1 latency ({len(windows)} windows across {len(channels)} channel(s)): "
          f"{p1_latency_pooled * 1000:.1f} ms")

    fig_pooled = plt.figure(figsize=FIGSIZE)
    if stim_dur is not None:
        plt.axvspan(0, stim_dur, color="0.6", alpha=0.2, zorder=0, label="Stimulus")
    for w in windows[:20]:
        plt.plot(time, w["eeg_y"], color="0.8", linewidth=0.8, alpha=0.8)
    plt.plot(time, avg_pooled, color="tab:blue", linewidth=2, label="Average ERP")
    plt.fill_between(time, avg_pooled - std_pooled, avg_pooled + std_pooled,
                     color="tab:blue", alpha=0.2, label="±1 SD")
    plt.axvline(0, color="0.0", linestyle="--", label="Trigger onset")
    plt.axvline(p1_latency_pooled, color="tab:red", linestyle="--",
               label=f"P1 latency ({p1_latency_pooled * 1000:.1f} ms)")
    plt.xlabel("Peri-stimulus time (s)")
    plt.ylabel("EEG amplitude (µV)")
    plt.title("Grand average ERP (all channels pooled)")
    plt.legend(loc="upper right")
    save_pgf(fig_pooled, "erp_latency_pooled")

    return p1_latency


def plot_erp_by_subject(subject_curves: list[dict]) -> None:
    """Overlay one averaged ERP curve per subject, all centred on stimulus
    onset, with the stimulus window (as in plot_erp_latency) and each
    subject's P1 peak marked. ``subject_curves`` is a list of {"name",
    "windows", "fs"}, where "windows" are get_erp_windows() epochs (all for
    the same channel) for that subject and "fs" is their sampling rate.
    Curves are left unlabelled in the legend (only the stimulus markers are)
    -- per-subject P1 latencies are printed instead."""
    fig = plt.figure(figsize=FIGSIZE)

    all_windows = [w for c in subject_curves for w in c["windows"]]
    stim_durs   = [w["stim_dur_s"] for w in all_windows if w["stim_dur_s"] is not None]
    stim_dur    = np.mean(stim_durs) if stim_durs else None
    if stim_dur is not None:
        plt.axvspan(0, stim_dur, color="0.6", alpha=0.2, zorder=0, label="Stimulus")

    p1_latencies = []
    for c in subject_curves:
        fs   = c["fs"]
        time = c["windows"][0]["time_s"] - c["windows"][0]["time_s"][int(0.25 * fs)]
        avg  = np.mean([w["eeg_y"] for w in c["windows"]], axis=0)
        line, = plt.plot(time, avg, linewidth=1.5, alpha=0.8)

        post_stim_mask = (time >= 0.02) & (time <= 0.100)
        window_idx     = np.where(post_stim_mask)[0]
        p1_idx         = window_idx[np.argmax(avg[window_idx])]
        p1_latency     = time[p1_idx]
        plt.plot(p1_latency, avg[p1_idx], "o", color=line.get_color(), zorder=5)
        print(f"{c['name']}: P1 latency {p1_latency * 1000:.1f} ms")
        p1_latencies.append(p1_latency)

    print(f"Mean P1 latency across {len(p1_latencies)} subject(s): "
          f"{np.mean(p1_latencies) * 1000:.1f} ms")
    print(f"Std P1 latency across {len(p1_latencies)} subject(s): "
          f"{np.std(p1_latencies) * 1000:.1f} ms")

    plt.axvline(0, color="0.0", linestyle="--", label="Stimulus onset")
    plt.xlabel("Time [s]")
    plt.ylabel(r"Amplitude [$\mu$V]")
    plt.legend(loc="upper right")

    plt.tight_layout()

    save_pdf(fig, "erp_by_subject")
    save_pgf(fig, "erp_by_subject")
