"""
ecHT vs. c-ecHT vs. offline Hilbert: cosine-based simulations.

Panels:

  a) SNR sweep: |error| against the true phase
  b) SNR sweep: signed difference c-ecHT - offline HT
  c) SNR sweep: signed error c-ecHT - true phase
  d) SNR sweep: signed error offline HT - true phase

Each SNR point uses N_PHASE_SAMPLES evenly spaced initial phases with
N_NOISE_TRIALS noise realisations each, so the statistics average over
phase (harmonic_experiments.py uses phi0 = 0 for all SNR trials).

The SNR is narrow-band: signal power over the power of the white noise
inside [f0 - SNR_BW/2, f0 + SNR_BW/2] (default [0.7, 1.3] f0). For white
noise this is the broadband SNR + 10 log10(SFREQ / (2 SNR_BW)), i.e.
+13.3 dB for SNR_BW = 6 Hz at 256 Hz.

The signed difference is computed per trial (same signal, same noise
realisation), so it shows how far and in which direction the causal c-ecHT
estimate deviates from the offline Hilbert reference. Panels c and d show
the signed error of c-ecHT and offline HT against the true phase. Panels
b-d show the circular mean ± circular SD (shaded) and, for comparison, the
ordinary (linear) mean ± SD of the wrapped differences in [-180, 180] deg
(dash-dotted). The two agree for small spread; at low SNR the linear SD
saturates (uniform phase: 180/sqrt(3) ~ 104 deg) while the circular SD
keeps growing.

Fixed conditions: Butterworth band-pass centred on f0, bandwidth BW and
order FILT_ORDER for ecHT / c-ecHT, bandwidth OFFLINE_BW and order
OFFLINE_FILT_ORDER for the offline Hilbert (defaults: [0.7, 1.3] f0 for
both, order 1 vs. 4 as in analysis/core.py); no frequency drift.

All three methods estimate the phase at the same target sample k:

  - ecHT / c-ecHT (causal): input is the W samples ending at k,
    x[k-W+1 .. k]; the estimate is the endpoint z[-1].
  - offline Hilbert (non-causal reference): zero-phase band-pass
    (sosfiltfilt, OFFLINE_BW / OFFLINE_FILT_ORDER) followed by scipy.signal.hilbert on
    the symmetric window x[k-L .. k+L]; the estimate is the centre sample.
    Edge effects of the FFT-based Hilbert transform and of filtfilt are
    thus pushed L samples away from the target.

L is fixed to OFFLINE_HALF_CYCLES (8 cycles of 1/f0), long enough for the
offline reference to have converged (~0.1 deg on a clean cosine with the
order-4 band-pass).
Set OFFLINE_HALF_CYCLES = None for L = W-1 (offline window = past + future
of the causal window).

Note that filtfilt applies the band-pass twice (|H|^2), i.e. the effective
order of the offline pipeline is 2 * OFFLINE_FILT_ORDER.

The signal time axis starts at the first sample of the causal window, so
for the causal methods the setup is identical to harmonic_experiments.py
(phi0 = phase at x[k-W+1]); the offline method additionally sees samples
after k.
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import circmean, circstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cecHT_ext"))
import _bootstrap  # noqa: E402,F401  -> puts the cecHT submodule on sys.path
from phase import ECHT  # noqa: E402
from utils import _wrap_phase  # noqa: E402

# Global parameters
F0 = 10                # central frequency in Hz
SFREQ = 256            # sampling frequency in Hz
BW = F0 * 0.6          # ecHT / c-ecHT band-pass bandwidth (Hz), centred on f0
FILT_ORDER = 2         # ecHT / c-ecHT Butterworth order (scipy band-pass convention)
OFFLINE_BW = F0 * 0.6  # offline HT band-pass bandwidth (Hz), as analysis/core.py
OFFLINE_FILT_ORDER = 4 # offline HT Butterworth order (as analysis/core.py)

N_PHASE_SAMPLES = 5            # initial phases per SNR point (evenly spaced)
N_NOISE_TRIALS = int(3e3)       # noise trials per initial phase
SNR_STEPS = 50                  # steps of the SNR range
SNR_BW = F0 * 0.4               # band around f0 in which the SNR is defined (Hz)
N_CYCLES = 2.1                  # causal window length (cycles of 1/f0)

# Half-length of the offline Hilbert window in cycles of 1/f0.
# None -> L = W - 1 (symmetric extension of the causal window).
OFFLINE_HALF_CYCLES = 64

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Figure style
blue = "#4E79A7"
orange = "#F28E2B"
green = "#59A14F"
gray = "#BAB0AC"

METHODS = ("echt", "cecht", "offline")
LABELS = {"echt": "ecHT", "cecht": "c-ecHT", "offline": "offline HT"}
COLORS = {"echt": orange, "cecht": blue, "offline": green}
LINESTYLES = {"echt": "-", "cecht": "--", "offline": ":"}
LIN_COLOR = "0.25"      # linear (non-circular) mean ± SD in the signed panels
LIN_LINESTYLE = "-."

SIGNED_KEYS = ("diff", "signed_cecht", "signed_offline")

FONT_SIZE = 20
TICK_SIZE = 0.75 * FONT_SIZE
LINE_WIDTH = 1.5
AXIS_LINKS = 1.0        # Thickness of plot spines


def set_mpl_style():
    """Global Matplotlib style"""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": AXIS_LINKS,
        "legend.fontsize": FONT_SIZE - 2,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "lines.linewidth": LINE_WIDTH,
        "figure.dpi": 300,
        "text.usetex": True,
    })


def add_panel_label(ax, label, text):
    """Panel label in the same style as intro diagram, e.g. '(A) Noise robustness'."""
    ax.text(
        0.02,
        1.02,
        rf"\textbf{{{label}}} {text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )


# Utility functions
def angle_diff(phi1, phi2):
    """Smallest signed difference phi1 - phi2 in [-pi, pi]."""
    return _wrap_phase(phi1 - phi2)

def window_lengths(n_cycles):
    """Causal window length W and offline half-length L (samples)."""
    W = int(np.round(n_cycles / F0 * SFREQ))
    if OFFLINE_HALF_CYCLES is None:
        L = W - 1
    else:
        L = int(np.round(OFFLINE_HALF_CYCLES / F0 * SFREQ))
    return W, L


def generate_trial(n_cycles, phi0=0):
    """
    Generate x[n] = cos(2 pi f0 t + phi0), long enough for both the causal
    window and the offline window.

    t = 0 at the first sample of the causal window, i.e. the causal input
    is identical to generate_cosine_window() in harmonic_experiments.py.

    Returns
    -------
    x : ndarray
        Clean cosine samples.
    k : int
        Index of the target sample (end of the causal window).
    W : int
        Causal window length (samples).
    L : int
        Offline window half-length (samples).
    true_phase : float
        True phase at sample k (rad).
    """
    W, L = window_lengths(n_cycles)
    k = max(W - 1, L)          # samples before the target
    n = np.arange(k + 1 + L)
    t = (n - (k - W + 1)) / SFREQ
    phase = 2 * np.pi * F0 * t + phi0
    return np.cos(phase), k, W, L, _wrap_phase(phase[k])


L_FREQ = F0 - BW / 2
H_FREQ = F0 + BW / 2
SOS = butter(OFFLINE_FILT_ORDER, [F0 - OFFLINE_BW / 2, F0 + OFFLINE_BW / 2],
             btype="band", fs=SFREQ, output="sos")


def endpoint_phase_echt(x, calibrate):
    # new instance per call: n_fft and the calibration gain depend on len(x)
    echt = ECHT(
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        sfreq=SFREQ,
        filt_order=FILT_ORDER,
        filter_type="butter",
        calibrate=calibrate,
        f0=(F0 if calibrate else None),
    )
    z = echt.fit_transform(x).ravel()
    return np.angle(z[-1])


def center_phase_offline(x):
    """Zero-phase band-pass + Hilbert, phase at the centre sample."""
    # scipy's default padlen, capped for very short windows
    padlen = min(3 * (2 * len(SOS) + 1), len(x) - 1)
    z = hilbert(sosfiltfilt(SOS, x, padlen=padlen))
    return np.angle(z[len(x) // 2])


def estimate_phases(x, k, W, L):
    """Phase at sample k for all three methods (rad)."""
    x_causal = x[k - W + 1 : k + 1]
    return {
        "echt": endpoint_phase_echt(x_causal, calibrate=False),
        "cecht": endpoint_phase_echt(x_causal, calibrate=True),
        "offline": center_phase_offline(x[k - L : k + L + 1]),
    }


# SNR sweep
def _snr_one_point(args):
    snr_db, seed = args
    rng = np.random.default_rng(seed)
    snr_lin = 10 ** (snr_db / 10)

    n_total = N_PHASE_SAMPLES * N_NOISE_TRIALS
    errs = {m: np.empty(n_total) for m in METHODS}
    # signed phase differences (rad): c-ecHT - offline HT, method - true phase
    signed = {key: np.empty(n_total) for key in SIGNED_KEYS}
    t = 0
    # evenly spaced initial phases, so the statistics average over phase
    # instead of reflecting the error at one particular phase
    for i in range(N_PHASE_SAMPLES):
        phi0 = i / N_PHASE_SAMPLES * 2 * np.pi
        x_clean, k, W, L, true_phase = generate_trial(N_CYCLES, phi0)
        # signal power over the causal window, as in harmonic_experiments.py
        signal_power = np.mean(x_clean[k - W + 1 : k + 1] ** 2)
        # narrow-band SNR: white noise of variance s^2 has power
        # s^2 * SNR_BW / (SFREQ / 2) inside [f0 - SNR_BW/2, f0 + SNR_BW/2]
        noise_std = np.sqrt(signal_power / snr_lin * SFREQ / (2 * SNR_BW))
        for _ in range(N_NOISE_TRIALS):
            # one noise realisation shared by all methods (offline also sees
            # the noisy future samples)
            x_noisy = x_clean + rng.normal(0, noise_std, size=x_clean.shape)
            phis = estimate_phases(x_noisy, k, W, L)
            for m in METHODS:
                errs[m][t] = np.degrees(np.abs(angle_diff(phis[m], true_phase)))
            signed["diff"][t] = angle_diff(phis["cecht"], phis["offline"])
            signed["signed_cecht"][t] = angle_diff(phis["cecht"], true_phase)
            signed["signed_offline"][t] = angle_diff(phis["offline"], true_phase)
            t += 1

    out = {m: (float(errs[m].mean()), float(errs[m].std(ddof=0))) for m in METHODS}
    for key, a in signed.items():
        out[key] = (
            float(np.degrees(circmean(a, high=np.pi, low=-np.pi))),
            float(np.degrees(circstd(a, high=np.pi, low=-np.pi))),
        )
        # ordinary mean / SD of the wrapped differences
        out[f"{key}_lin"] = (
            float(np.degrees(a.mean())),
            float(np.degrees(a.std(ddof=0))),
        )
    return out


def compute_snr_sweep(seed, print_every=10):
    rng = np.random.default_rng(seed)
    snr_db_list = np.linspace(-20, 20, SNR_STEPS)

    per_point_seeds = rng.integers(0, 2**32 - 1, size=len(snr_db_list))
    task_args = [
        (snr_db, int(per_point_seeds[i])) for i, snr_db in enumerate(snr_db_list)
    ]

    total = len(task_args)
    results = [None] * total
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_snr_one_point, a): i for i, a in enumerate(task_args)}
        for completed, fut in enumerate(as_completed(futures), start=1):
            results[futures[fut]] = fut.result()
            if completed % print_every == 0 or completed == total:
                print(f"  SNR sweep: {completed}/{total} points done", flush=True)

    stats = {
        m: {
            "mean": np.array([r[m][0] for r in results]),
            "std": np.array([r[m][1] for r in results]),
        }
        for m in (*METHODS, *SIGNED_KEYS, *(f"{k}_lin" for k in SIGNED_KEYS))
    }
    return {"snr_db": snr_db_list, "stats": stats}


DEG_FORMATTER = mpl.ticker.FuncFormatter(lambda v, _: rf"${v:g}^\circ$")


def plot_snr_sweep(ax, r):
    snr_db, stats = r["snr_db"], r["stats"]
    eps = 1e-6
    for m in METHODS:
        mu, sd = stats[m]["mean"], stats[m]["std"]
        ax.semilogy(snr_db, mu, LINESTYLES[m], color=COLORS[m],
                    label=LABELS[m], linewidth=LINE_WIDTH)
        ax.fill_between(snr_db, np.clip(mu - sd, eps, None), mu + sd,
                        color=COLORS[m], alpha=0.18)

    ax.set_xlabel(r"narrow-band SNR [dB]")
    ax.set_ylabel(r"$|$phase error$|$ $[^\circ]$")
    lo = min(stats[m]["mean"].min() for m in METHODS)
    hi = max(stats[m]["mean"].max() for m in METHODS)
    ax.set_ylim(0.8 * lo, 1.2 * hi)
    # labelled ticks at 1-2-5 per decade, unlabelled minor ticks in between
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(1, 10)))
    ax.yaxis.set_major_formatter(DEG_FORMATTER)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid(True, linestyle=":", which="both", axis="both")
    ax.set_axisbelow(True)

    
def plot_snr_signed(ax, r, key, method, ylabel):
    """Circular (shaded) and linear (dash-dotted) mean ± SD of a signed phase
    difference vs. SNR."""
    snr_db = r["snr_db"]
    mu, sd = r["stats"][key]["mean"], r["stats"][key]["std"]
    ax.plot(snr_db, mu, LINESTYLES[method], color=COLORS[method],
            linewidth=LINE_WIDTH)
    ax.fill_between(snr_db, mu - sd, mu + sd, color=COLORS[method], alpha=0.18)

    mu, sd = r["stats"][f"{key}_lin"]["mean"], r["stats"][f"{key}_lin"]["std"]
    ax.plot(snr_db, mu, LIN_LINESTYLE, color=LIN_COLOR, linewidth=LINE_WIDTH)
    for edge in (mu - sd, mu + sd):
        ax.plot(snr_db, edge, LIN_LINESTYLE, color=LIN_COLOR,
                linewidth=0.6 * LINE_WIDTH, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, zorder=1)

    ax.set_xlabel(r"narrow-band SNR [dB]")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=12, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(DEG_FORMATTER)
    ax.grid(True, linestyle=":", which="both", axis="y")
    ax.grid(True, linestyle=":", which="major", axis="x")
    ax.set_axisbelow(True)


def main(output_prefix="ecHT_offline_ht_simulations"):
    set_mpl_style()
    seed = 0

    res_snr = compute_snr_sweep(seed)

    fig, ax = plt.subplots(2, 2, figsize=(12, 13))
    ax = ax.ravel()

    plot_snr_sweep(ax[0], res_snr)
    plot_snr_signed(ax[1], res_snr, "diff", "cecht",
                    r"c-ecHT $-$ offline HT $[^\circ]$")
    plot_snr_signed(ax[2], res_snr, "signed_cecht", "cecht",
                    r"c-ecHT $-$ true phase $[^\circ]$")
    plot_snr_signed(ax[3], res_snr, "signed_offline", "offline",
                    r"offline HT $-$ true phase $[^\circ]$")
    # same y-range for c and d so the two errors are directly comparable
    ax[2].sharey(ax[3])
    ax[3].autoscale(axis="y")

    add_panel_label(ax[0], "a", "Noise robustness")
    add_panel_label(ax[1], "b", r"c-ecHT vs. offline HT")
    add_panel_label(ax[2], "c", r"c-ecHT vs. true phase")
    add_panel_label(ax[3], "d", r"offline HT vs. true phase")

    # Shared legend
    handles = [
        mlines.Line2D([], [], color=COLORS[m], linestyle=LINESTYLES[m],
                      label=LABELS[m])
        for m in METHODS
    ]
    handles.append(mpatches.Patch(
        facecolor=gray, alpha=0.18, edgecolor="k", linewidth=0.5,
        label=r"$\pm 1$ SD (circular in b--d)",
    ))
    handles.append(mlines.Line2D(
        [], [], color=LIN_COLOR, linestyle=LIN_LINESTYLE,
        label=r"linear mean $\pm 1$ SD (b--d)",
    ))
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=True,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("0.5")

    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])

    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figdir / f"{output_prefix}.pdf", bbox_inches="tight")

    # results per SNR (deg): mean |error| of c-ecHT, circular means of the signed differences
    stats = res_snr["stats"]
    table = np.column_stack([
        res_snr["snr_db"],
        stats["cecht"]["mean"],
        stats["diff"]["mean"],
        stats["signed_cecht"]["mean"],
        stats["signed_offline"]["mean"],
    ])
    np.savetxt(
        figdir / f"{output_prefix}.csv", table, delimiter=",", fmt="%.6g", comments="",
        header="snr_db,cecht_abs_error,cecht_minus_ht,cecht_minus_gt,ht_minus_gt",
    )


if __name__ == "__main__":
    main()
