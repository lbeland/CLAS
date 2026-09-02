"""
ecHT harmonic experiments: cosine-based simulations.

Panels:

  1) Bandwidth sweep
  2) Filter order sweep
  3) Frequency drift
  4) SNR sweep
  5) Filter-type comparison (Butterworth, Bessel, Cheby I/II, Elliptic)
  6) Window length sweep (wide range, non-integer cycles).

For all sweep-type plots we show:
  - mean absolute phase error (deg)
  - ±1 std of |error| as a shaded band around the mean

Both variants are always calibrated (c-ecHT). We compare the internal
forward-transform algorithm instead: c-ecHT with a batch FFT (displayed in
orange) vs. c-ecHT with a per-sample sliding DFT (displayed in blue).
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------------
# ECHTExt = pristine phase.ECHT + the sliding-DFT forward transform
# (transform_sdft). See ../echt_ext.py.
# ---------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401  -> puts the cecHT submodule on sys.path
from echt_ext import ECHTExt as ECHT  # noqa: E402

# ---------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------
F0 = 10                # central frequency in Hz
SFREQ = 256            # sampling frequency in Hz (for most experiments)
BW_DEFAULT = F0 * 0.3  # default bandwidth (Hz)

N_PHASE_SAMPLES = 90            # Samples of initial phase
N_NOISE_TRIALS = int(1e4)       # Monte-Carlo trials for noise
N_CYCLES = 2.1
SMOOTH_WINDOW_CYCLES = 0
N_WINDOW_STEPS = 100

N_WORKERS = min(9, max(1, (os.cpu_count() or 2) - 1))

MASTER_RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------
blue = "#4E79A7"
orange = "#F28E2B"
red = "#E15759"
green = "#59A14F"
yellow = "#EDC948"
purple = "#B07AA1"
teal = "#76B7B2"
gray = "#BAB0AC"
global_alpha = 0.6

COL_FFT = orange
COL_SDFT = blue

def set_mpl_style():
    """Global Matplotlib style"""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.1,
        "figure.dpi": 300,
        "text.usetex": False,
    })


COMMON_BBOX = dict(
    facecolor="white",
    edgecolor=gray,
    boxstyle="round,pad=0.2",
    alpha=0.9,
)


def add_panel_label(ax, label, text):
    """
    Panel label in the same style as intro diagram, e.g. '(A) Window length'.
    """
    ax.text(
        0.02,
        1.02,
        rf"\textbf{{({label})}} {text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        # bbox=COMMON_BBOX,
    )

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def wrap_phase(phi):
    """Wrap phase to [-pi, pi]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def angle_diff(phi1, phi2):
    """Smallest signed difference phi1 - phi2 in [-pi, pi]."""
    return wrap_phase(phi1 - phi2)


def generate_cosine_window(n_cycles, f0, sfreq, phi0=0.0):
    """
    Generate x[n] = cos(2π f0 t + phi0) with a given number of cycles.

    Returns
    -------
    t : ndarray
        Time vector (s).
    x : ndarray
        Cosine samples.
    true_phase_end : float
        True underlying phase at the last sample (rad).
    """
    duration = n_cycles / f0
    n_samples = int(np.round(duration * sfreq))
    # transform_sdft requires an even window length (jurihock/sdft's sliding
    # window is always 2*dftsize samples internally); round up so the FFT
    # and sliding-DFT variants can be compared on the exact same window.
    n_samples += n_samples % 2
    t = np.arange(n_samples) / sfreq
    phase = 2 * np.pi * f0 * t + phi0
    x = np.cos(phase)
    true_phase_end = wrap_phase(phase[-1])
    return t, x, true_phase_end


def endpoint_phase_echt(
    x,
    f0,
    sfreq,
    bw=None,
    order=1,
    filter_type="butter",
    method="fft",
):
    """Endpoint phase using calibrated ECHT, via either the batch-FFT
    (method='fft') or sliding-DFT (method='sdft') forward transform."""
    if bw is None:
        bw = BW_DEFAULT
    if method not in ("fft", "sdft"):
        raise ValueError("method must be 'fft' or 'sdft'")
    l_freq = f0 - bw / 2.0
    h_freq = f0 + bw / 2.0
    echt = ECHT(
        l_freq=l_freq,
        h_freq=h_freq,
        sfreq=sfreq,
        fft_mode="exact",  # n_fft == n_samples exactly, identical grid for both methods
        filt_order=order,
        filter_type=filter_type,
        calibrate=True,
        f0=f0,
    )
    echt.fit(x)
    if method == "fft":
        z = echt.transform(x).ravel()
    else:
        z = echt.transform_sdft(x).ravel()
    return np.angle(z[-1])


# ---------------------------------------------------------------------
# Window length sweep
# ---------------------------------------------------------------------
def compute_window_length_sweep():
    """
    Window-length sweep summarized as violin plots over ±0.5 cycles.

    For each central cycle count c in {1,2,3,4} we draw many non-integer
    window lengths n_cycles in [c-0.5, c+0.5], generate signals, and
    aggregate the resulting phase errors into one distribution per band.
    """
    cycles = np.array([1, 2, 3, 4])
    fft_err = []
    sdft_err = []

    window_span = 0.5
    n_window_steps = N_WINDOW_STEPS if "N_WINDOW_STEPS" in globals() else 25

    for c in cycles:
        # Sample window lengths uniformly in [c-0.5, c+0.5]
        lo = max(c - window_span, 0.2)  # avoid ridiculously short windows
        hi = c + window_span
        n_cycles_vals = np.linspace(lo, hi, n_window_steps)

        errs_fft_band = []
        errs_sdft_band = []

        for n_cyc in n_cycles_vals:
            for i in range(N_PHASE_SAMPLES):
                phi0 = i / N_PHASE_SAMPLES * 2 * np.pi

                # Non-integer number of cycles here
                _, x, true_phase_end = generate_cosine_window(
                    n_cyc, F0, SFREQ, phi0
                )

                phi_fft = endpoint_phase_echt(
                    x, F0, SFREQ, method="fft"
                )
                phi_sdft = endpoint_phase_echt(
                    x, F0, SFREQ, method="sdft"
                )

                err_fft = np.degrees(
                    (angle_diff(phi_fft, true_phase_end))
                )
                err_sdft = np.degrees(
                    (angle_diff(phi_sdft, true_phase_end))
                )

                errs_fft_band.append(err_fft)
                errs_sdft_band.append(err_sdft)

        # One big distribution per cycle band
        fft_err.append(errs_fft_band)
        sdft_err.append(errs_sdft_band)

    return {
        "cycles": cycles,
        "fft": fft_err,
        "sdft": sdft_err,
    }


def plot_window_length_sweep(ax, r):
    cycles = r["cycles"]
    fft_err = [np.abs(u) for u in r["fft"]]
    sdft_err = [np.abs(ca) for ca in r["sdft"]]

    positions = np.arange(len(cycles))

    # pad each distribution with 0° and 180°
    fft_err = [np.concatenate([u, [0, 180]]) for u in fft_err]
    sdft_err = [np.concatenate([ca, [0, 180]]) for ca in sdft_err]

    v_fft = ax.violinplot(
        fft_err,
        positions=positions,
        widths=0.8,
        points=1000,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    v_sdft = ax.violinplot(
        sdft_err,
        positions=positions,
        widths=0.8,
        points=1000,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    # half-violin
    for i, pos in enumerate(positions):
        for body, side, color in [
            (v_fft["bodies"][i], "left", COL_FFT),
            (v_sdft["bodies"][i], "right", COL_SDFT),
        ]:
            path = body.get_paths()[0]
            verts = path.vertices
            xs = verts[:, 0]

            if side == "left":
                verts[:, 0] = np.minimum(xs, pos)
            else:
                verts[:, 0] = np.maximum(xs, pos)

            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(global_alpha)


    # median of |error| to display in violins
    median_fft = [np.median(u) for u in fft_err]
    median_sdft = [np.median(ca) for ca in sdft_err]

    ax.scatter(positions - 0.1, median_fft, marker=">", s=20,
               color=COL_FFT, zorder=3, label="FFT",
               edgecolor="black", linewidth=0.75)
    ax.scatter(positions + 0.1, median_sdft, marker="<", s=20,
               color=COL_SDFT, zorder=3, label="sliding DFT",
               edgecolor="black", linewidth=0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels([rf"{c}\,$\pm$\,0.5" for c in cycles])
    ax.set_xlabel(r"cycles of $1/f_0$")
    ax.set_ylim(0, 45)
    ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels([r"$10^\circ$", r"$20^\circ$", r"$30^\circ$", r"$40^\circ$"])
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)



# ---------------------------------------------------------------------
# Bandwidth sweep
# ---------------------------------------------------------------------
def compute_bandwidth_sweep():
    bw_factors = np.linspace(0.1, 1, 100)
    nyq = SFREQ / 2.0

    valid_f = []
    mean_fft = []
    std_fft = []
    max_fft = []
    mean_sdft = []
    std_sdft = []
    max_sdft = []

    n_cycles = N_CYCLES

    for f in bw_factors:
        bw = f * F0
        l_freq = F0 - bw / 2.0
        h_freq = F0 + bw / 2.0
        if l_freq <= 0.0 or h_freq >= nyq:
            continue

        errs_fft = []
        errs_sdft = []
        for i in range(N_PHASE_SAMPLES):
            phi0 = i/N_PHASE_SAMPLES * 2*np.pi
            _, x, true_phase_end = generate_cosine_window(
                n_cycles, F0, SFREQ, phi0
            )
            phi_fft = endpoint_phase_echt(
                x, F0, SFREQ, bw=bw, order=2, method="fft"
            )
            phi_sdft = endpoint_phase_echt(
                x, F0, SFREQ, bw=bw, order=2, method="sdft"
            )
            err_fft = np.degrees(np.abs(angle_diff(phi_fft, true_phase_end)))
            err_sdft = np.degrees(np.abs(angle_diff(phi_sdft, true_phase_end)))
            errs_fft.append(err_fft)
            errs_sdft.append(err_sdft)

        errs_fft = np.asarray(errs_fft)
        errs_sdft = np.asarray(errs_sdft)

        valid_f.append(f)
        mean_fft.append(float(errs_fft.mean()))
        std_fft.append(float(errs_fft.std(ddof=0)))
        max_fft.append(float(errs_fft.max()))
        mean_sdft.append(float(errs_sdft.mean()))
        std_sdft.append(float(errs_sdft.std(ddof=0)))
        max_sdft.append(float(errs_sdft.max()))

    return {
        "bw_factors": np.array(valid_f),
        "mean_fft": np.array(mean_fft),
        "std_fft": np.array(std_fft),
        "max_fft": np.array(max_fft),
        "mean_sdft": np.array(mean_sdft),
        "std_sdft": np.array(std_sdft),
        "max_sdft": np.array(max_sdft),
    }


def plot_bandwidth_sweep(ax, r):
    f = r["bw_factors"]
    mean_fft, std_fft, max_fft = r["mean_fft"], r["std_fft"], r["max_fft"]
    mean_sdft, std_sdft, max_sdft = r["mean_sdft"], r["std_sdft"], r["max_sdft"]

    line_mu_fft, = ax.plot(
        f, mean_fft, "-", label="FFT", color=COL_FFT
    )
    line_mu_sdft, = ax.plot(
        f, mean_sdft, "--", label="sliding DFT", color=COL_SDFT
    )

    lower_fft = np.clip(mean_fft - std_fft, 0, None)
    upper_fft = mean_fft + std_fft
    ax.fill_between(
        f, lower_fft, upper_fft,
        color=line_mu_fft.get_color(), alpha=0.18,
    )

    lower_sdft = np.clip(mean_sdft - std_sdft, 0, None)
    upper_sdft = mean_sdft + std_sdft
    ax.fill_between(
        f, lower_sdft, upper_sdft,
        color=line_mu_sdft.get_color(), alpha=0.18,
    )

    ax.set_xlabel(r"Bandwidth / $f_0$")
    ax.set_ylabel(r"$|$Phase error$|$ $[^\circ]$")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5, alpha=0.7)


# ---------------------------------------------------------------------
# Filter order sweep
# ---------------------------------------------------------------------
def compute_order_sweep():
    orders = np.array([1, 2, 3, 4, 5])

    mean_fft = []
    std_fft = []
    max_fft = []
    mean_sdft = []
    std_sdft = []
    max_sdft = []

    n_cycles = N_CYCLES

    for order in orders:
        errs_fft = []
        errs_sdft = []
        for i in range(N_PHASE_SAMPLES):
            phi0 = i/N_PHASE_SAMPLES *2*np.pi
            _, x, true_phase_end = generate_cosine_window(
                n_cycles, F0, SFREQ, phi0
            )
            phi_fft = endpoint_phase_echt(
                x, F0, SFREQ, bw=BW_DEFAULT, order=order, method="fft"
            )
            phi_sdft = endpoint_phase_echt(
                x, F0, SFREQ, bw=BW_DEFAULT, order=order, method="sdft"
            )
            err_fft = np.degrees(np.abs(angle_diff(phi_fft, true_phase_end)))
            err_sdft = np.degrees(np.abs(angle_diff(phi_sdft, true_phase_end)))
            errs_fft.append(err_fft)
            errs_sdft.append(err_sdft)

        errs_fft = np.asarray(errs_fft)
        errs_sdft = np.asarray(errs_sdft)

        mean_fft.append(float(errs_fft.mean()))
        std_fft.append(float(errs_fft.std(ddof=0)))
        max_fft.append(float(errs_fft.max()))
        mean_sdft.append(float(errs_sdft.mean()))
        std_sdft.append(float(errs_sdft.std(ddof=0)))
        max_sdft.append(float(errs_sdft.max()))

    return {
        "orders": orders,
        "mean_fft": np.array(mean_fft),
        "std_fft": np.array(std_fft),
        "max_fft": np.array(max_fft),
        "mean_sdft": np.array(mean_sdft),
        "std_sdft": np.array(std_sdft),
        "max_sdft": np.array(max_sdft),
    }


def plot_order_sweep(ax, r):
    orders = r["orders"]
    mean_fft, std_fft, max_fft = r["mean_fft"], r["std_fft"], r["max_fft"]
    mean_sdft, std_sdft, max_sdft = r["mean_sdft"], r["std_sdft"], r["max_sdft"]

    line_mu_fft, = ax.plot(
        orders, mean_fft, "-", label="FFT", color=COL_FFT
    )
    line_mu_sdft, = ax.plot(
        orders, mean_sdft, "--", label="sliding DFT", color=COL_SDFT
    )

    lower_fft = np.clip(mean_fft - std_fft, 0, None)
    upper_fft = mean_fft + std_fft
    ax.fill_between(
        orders, lower_fft, upper_fft,
        color=line_mu_fft.get_color(), alpha=0.18,
    )

    lower_sdft = np.clip(mean_sdft - std_sdft, 0, None)
    upper_sdft = mean_sdft + std_sdft
    ax.fill_between(
        orders, lower_sdft, upper_sdft,
        color=line_mu_sdft.get_color(), alpha=0.18,
    )

    ax.set_xlabel("Filter order (Butterworth)")
    ax.set_xticks(orders)
    ax.set_xticklabels(2 * orders)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5, alpha=0.7)



# ---------------------------------------------------------------------
# SNR sweep
# ---------------------------------------------------------------------
def compute_snr_sweep(seed):
    rng = np.random.default_rng(seed)
    snr_db_list = np.linspace(-10, 20, 1000)

    n_cycles = N_CYCLES
    _, x_clean, true_phase_end = generate_cosine_window(
        n_cycles, F0, SFREQ, phi0=0.0
    )
    signal_power = np.mean(x_clean ** 2)

    mean_fft = []
    std_fft = []
    mean_sdft = []
    std_sdft = []

    for snr_db in snr_db_list:
        snr_lin = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_lin
        noise_std = np.sqrt(noise_power)

        errs_fft = []
        errs_sdft = []
        for _ in range(N_NOISE_TRIALS):
            noise = rng.normal(0.0, noise_std, size=x_clean.shape)
            x_noisy = x_clean + noise
            phi_fft = endpoint_phase_echt(
                x_noisy, F0, SFREQ, bw=BW_DEFAULT, order=2, method="fft"
            )
            phi_sdft = endpoint_phase_echt(
                x_noisy, F0, SFREQ, bw=BW_DEFAULT, order=2, method="sdft"
            )
            err_fft = np.degrees(np.abs(angle_diff(phi_fft, true_phase_end)))
            err_sdft = np.degrees(np.abs(angle_diff(phi_sdft, true_phase_end)))
            errs_fft.append(err_fft)
            errs_sdft.append(err_sdft)

        errs_fft = np.asarray(errs_fft)
        errs_sdft = np.asarray(errs_sdft)

        mean_fft.append(float(errs_fft.mean()))
        std_fft.append(float(errs_fft.std(ddof=0)))
        mean_sdft.append(float(errs_sdft.mean()))
        std_sdft.append(float(errs_sdft.std(ddof=0)))

    return {
        "snr_db": snr_db_list,
        "mean_fft": np.array(mean_fft),
        "std_fft": np.array(std_fft),
        "mean_sdft": np.array(mean_sdft),
        "std_sdft": np.array(std_sdft),
    }


def plot_snr_sweep(ax, r):
    snr_db = r["snr_db"]
    mean_fft, std_fft = r["mean_fft"], r["std_fft"]
    mean_sdft, std_sdft = r["mean_sdft"], r["std_sdft"]

    line_mu_fft, = ax.semilogy(
        snr_db, mean_fft, "-", label="FFT", color=COL_FFT
    )
    line_mu_sdft, = ax.semilogy(
        snr_db, mean_sdft, "--", label="sliding DFT", color=COL_SDFT
    )

    eps = 1e-6
    lower_fft = np.clip(mean_fft - std_fft, eps, None)
    upper_fft = np.clip(mean_fft + std_fft, eps, None)
    ax.fill_between(
        snr_db, lower_fft, upper_fft,
        color=line_mu_fft.get_color(), alpha=0.18,
    )

    lower_sdft = np.clip(mean_sdft - std_sdft, eps, None)
    upper_sdft = np.clip(mean_sdft + std_sdft, eps, None)
    ax.fill_between(
        snr_db, lower_sdft, upper_sdft,
        color=line_mu_sdft.get_color(), alpha=0.18,
    )

    ax.set_xlabel(r"Input SNR (dB)")
    ax.set_ylabel(r"$|$phase error$|$ $[^\circ]$")
    ax.set_ylim(0.8 * min(mean_sdft), 1.2 * max(mean_fft))
    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5, alpha=0.7)


# ---------------------------------------------------------------------
# Chirp robustness
# ---------------------------------------------------------------------
def generate_detuned_cosine(n_cycles, f0, sfreq, delta_frac=0.0, phi0=0.0):
    duration = n_cycles / f0
    n_samples = int(round(duration * sfreq))
    n_samples += n_samples % 2  # transform_sdft requires an even window length
    t = np.arange(n_samples) / sfreq

    f_sig = f0 * (1.0 + delta_frac)
    phase = 2 * np.pi * f_sig * t + phi0
    x = np.cos(phase)
    true_phase_end = wrap_phase(phase[-1])
    return t, x, true_phase_end

def compute_chirp_robustness():
    delta_fracs = np.linspace(-0.1, 0.1, 100)
    n_cycles = N_CYCLES

    mean_fft = []; std_fft = []
    mean_sdft = []; std_sdft = []

    for df in delta_fracs:
        errs_fft = []
        errs_sdft = []
        for i in range(N_PHASE_SAMPLES):
            phi0 = i / N_PHASE_SAMPLES * 2 * np.pi
            _, x, true_phase_end = generate_detuned_cosine(
                n_cycles, F0, SFREQ, delta_frac=df, phi0=phi0
            )
            phi_fft = endpoint_phase_echt(x, F0, SFREQ, bw=BW_DEFAULT,
                                          order=2, method="fft")
            phi_sdft = endpoint_phase_echt(x, F0, SFREQ, bw=BW_DEFAULT,
                                          order=2, method="sdft")

            err_fft = np.degrees(np.abs(angle_diff(phi_fft, true_phase_end)))
            err_sdft = np.degrees(np.abs(angle_diff(phi_sdft, true_phase_end)))
            errs_fft.append(err_fft)
            errs_sdft.append(err_sdft)

        errs_fft = np.asarray(errs_fft)
        errs_sdft = np.asarray(errs_sdft)

        mean_fft.append(float(errs_fft.mean()))
        std_fft.append(float(errs_fft.std(ddof=0)))
        mean_sdft.append(float(errs_sdft.mean()))
        std_sdft.append(float(errs_sdft.std(ddof=0)))

    return {
        "delta_fracs": np.array(delta_fracs),
        "mean_fft": np.array(mean_fft),
        "std_fft": np.array(std_fft),
        "mean_sdft": np.array(mean_sdft),
        "std_sdft": np.array(std_sdft),
    }


def plot_chirp_robustness(ax, r):
    df = r["delta_fracs"]
    mean_fft, std_fft = r["mean_fft"], r["std_fft"]
    mean_sdft, std_sdft = r["mean_sdft"], r["std_sdft"]

    line_mu_fft, = ax.plot(
        df, mean_fft, "-", label="FFT", color=COL_FFT
    )
    line_mu_sdft, = ax.plot(
        df, mean_sdft, "--", label="sliding DFT", color=COL_SDFT
    )

    lower_fft = np.clip(mean_fft - std_fft, 0, None)
    upper_fft = mean_fft + std_fft
    ax.fill_between(
        df, lower_fft, upper_fft,
        color=line_mu_fft.get_color(), alpha=0.18,
    )

    lower_sdft = np.clip(mean_sdft - std_sdft, 0, None)
    upper_sdft = mean_sdft + std_sdft
    ax.fill_between(
        df, lower_sdft, upper_sdft,
        color=line_mu_sdft.get_color(), alpha=0.18,
    )

    ax.set_xlabel(r"$\Delta f/f_0$")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5, alpha=0.7)



# ---------------------------------------------------------------------
# Filter-type comparison
# ---------------------------------------------------------------------
def compute_filter_type():
    n_cycles = N_CYCLES
    order = 1
    bw = BW_DEFAULT

    filter_types = ["butter", "bessel", "cheby1", "cheby2", "ellip"]
    labels = ["Butter", "Bessel", "Cheby I", "Cheby II", "Cauer"]

    mean_fft = []
    std_fft = []
    mean_sdft = []
    std_sdft = []

    for ftype in filter_types:
        errs_rad_fft = []
        errs_rad_sdft = []

        for i in range(N_PHASE_SAMPLES):
            phi0 = i / N_PHASE_SAMPLES * 2 * np.pi
            _, x, true_phase_end = generate_cosine_window(
                n_cycles, F0, SFREQ, phi0
            )

            # calibrated ECHT with given filter type, batch FFT
            phi_fft = endpoint_phase_echt(
                x,
                F0,
                SFREQ,
                bw=bw,
                order=order,
                filter_type=ftype,
                method="fft",
            )
            # calibrated ECHT with given filter type, sliding DFT
            phi_sdft = endpoint_phase_echt(
                x,
                F0,
                SFREQ,
                bw=bw,
                order=order,
                filter_type=ftype,
                method="sdft",
            )

            err_fft = angle_diff(phi_fft, true_phase_end)
            err_sdft = angle_diff(phi_sdft, true_phase_end)
            errs_rad_fft.append(err_fft)
            errs_rad_sdft.append(err_sdft)

        errs_rad_fft = np.asarray(errs_rad_fft)
        errs_rad_sdft = np.asarray(errs_rad_sdft)

        errs_deg_fft = np.degrees(np.abs(errs_rad_fft))
        errs_deg_sdft = np.degrees(np.abs(errs_rad_sdft))

        mean_fft.append(float(errs_deg_fft.mean()))
        std_fft.append(float(errs_deg_fft.std(ddof=0)))
        mean_sdft.append(float(errs_deg_sdft.mean()))
        std_sdft.append(float(errs_deg_sdft.std(ddof=0)))

    return {
        "labels": labels,
        "mean_fft": np.array(mean_fft),
        "std_fft": np.array(std_fft),
        "mean_sdft": np.array(mean_sdft),
        "std_sdft": np.array(std_sdft),
    }


def plot_filter_type(ax, r):
    labels = np.array(r["labels"])
    mean_fft, std_fft = r["mean_fft"], r["std_fft"]
    mean_sdft, std_sdft = r["mean_sdft"], r["std_sdft"]

    # Console summary
    print("Filter-type comparison (calibrated ECHT, order=1, BW = f0/2):")
    print("{:<10} {:>12} {:>12} {:>12} {:>12}".format(
        "Filter", "mean|err| fft", "std fft",
        "mean|err| sdft", "std sdft"
    ))
    for lab, mu_u, cs_u, mu_c, cs_c in zip(
        labels, mean_fft, std_fft, mean_sdft, std_sdft
    ):
        print("{:<10} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
            lab, mu_u, cs_u, mu_c, cs_c
        ))
    print()

    # Identify Cheby II index
    outlier_name = "Cheby II"
    idx_out = np.where(labels == outlier_name)[0][0] if outlier_name in labels else None

    # y-limit from non–Cheby-II bars
    if idx_out is not None:
        mask_other = np.ones_like(labels, dtype=bool)
        mask_other[idx_out] = False
        y_max_others = float(
            max(
                (mean_fft[mask_other] + std_fft[mask_other]).max(),
                (mean_sdft[mask_other] + std_sdft[mask_other]).max(),
            )
        )
        y_lim = 1.05 * y_max_others
    else:
        y_lim = 1.05 * float(max(mean_fft.max(), mean_sdft.max()))

    x = np.arange(len(labels))
    width = 0.35

    face_fft = mcolors.to_rgba(COL_FFT, alpha=global_alpha)
    face_sdft = mcolors.to_rgba(COL_SDFT, alpha=global_alpha)

    bars_fft = ax.bar(
        x - width / 2,
        mean_fft,
        width,
        yerr=std_fft,
        capsize=2,
        label="FFT",
        color=face_fft,
        edgecolor="black",
        linewidth=1,
    )
    bars_sdft = ax.bar(
        x + width / 2,
        mean_sdft,
        width,
        yerr=std_sdft,
        capsize=2,
        label="sliding DFT",
        color=face_sdft,
        edgecolor="black",
        linewidth=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, y_lim)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    # Annotate Cheby II FFT-variant height on its clipped bar
    if idx_out is not None:
        bar_out = bars_fft[idx_out]
        x_out = bar_out.get_x() + 1.1*bar_out.get_width() / 2.0
        val_out = mean_fft[idx_out]

        ax.text(
            x_out,
            y_lim * 0.95,
            f"{val_out:.1f}°",
            ha="center",
            va="top",
            fontsize=7,
            rotation=90,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(output_prefix="ecHT_sdft_simulations"):
    set_mpl_style()

    # Prepare seeds for the six independent computations
    seed = MASTER_RNG.integers(0, 2**32 - 1, size=1)

    # Computations in parallel
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [
            ex.submit(compute_window_length_sweep),  # 0
            # ex.submit(compute_bandwidth_sweep),  # 1
            # ex.submit(compute_order_sweep),  # 2
            # ex.submit(compute_snr_sweep, seed),  # 3
            ex.submit(compute_chirp_robustness),  # 4
            # ex.submit(compute_filter_type),  # 5
        ]
        results = [f.result() for f in futures]

        (res_window,
        #  res_bandwidth,
        #  res_order,
        #  res_snr,
         res_chirp,
        #  res_filter
         ) = results

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))
    ax = axes.ravel()

    # plot_bandwidth_sweep(ax[0], res_bandwidth)  # Bandwidth
    # plot_order_sweep(ax[1], res_order)  # Filter order
    plot_chirp_robustness(ax[2], res_chirp)  # Frequency drift
    # plot_snr_sweep(ax[3], res_snr)  # Noise robustness
    # plot_filter_type(ax[4], res_filter)  # Filter type
    plot_window_length_sweep(ax[5], res_window)  # Window length

    # Panel labels in intro style
    # add_panel_label(ax[0], "A", "Bandwidth")
    # add_panel_label(ax[1], "B", "Filter order")
    add_panel_label(ax[2], "C", "Frequency drift")
    # add_panel_label(ax[3], "D", "Noise robustness")
    # add_panel_label(ax[4], "E", "Filter type")
    add_panel_label(ax[5], "F", "Window length")


    # Shared legend (FFT, sliding DFT, ±1 SD)
    line_fft = mlines.Line2D([], [], color=COL_FFT, linestyle="-", label="FFT")
    line_sdft = mlines.Line2D([], [], color=COL_SDFT, linestyle="--", label="sliding DFT")
    std_patch = mpatches.Patch(
        facecolor=gray,
        alpha=0.18,
        edgecolor="k",
        linewidth=0.5,
        label=r"$\pm 1$ SD",
    )

    legend = fig.legend(
        handles=[line_fft, line_sdft, std_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=3,
        frameon=True,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("0.5")

    fig.tight_layout(rect=[0.0, 0.10, 1.0, 1.0])

    _figdir = Path(__file__).resolve().parents[1] / "figures"
    fig.savefig(_figdir / f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(_figdir / f"{output_prefix}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
