import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, freqz_sos, bessel
from scipy import fftpack
import os

# Saving to a ".pgf" filename invokes the pgf backend automatically, so the
# default (interactive) backend stays active and plt.show() keeps working.
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

PLOTS_DIR = "/home/linda/Documents/MA/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

TEXTWIDTH    = 6.30045
ASPECT_RATIO = 3/4
FIG_WIDTH    = TEXTWIDTH
FIG_HEIGHT   = FIG_WIDTH * ASPECT_RATIO
FIGSIZE      = (FIG_WIDTH, FIG_HEIGHT)


def save_pgf(fig, name: str) -> None:
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.pgf"), bbox_inches="tight")

def save_png(fig, name: str) -> None:
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.png"), bbox_inches="tight", dpi=300)


def _dirichlet_kernel(alpha, N):
    """Length-N Dirichlet kernel D_N(alpha) = sin(N*alpha/2)/sin(alpha/2) * exp(i*alpha*(N-1)/2)."""
    alpha = np.asarray(alpha, dtype=float)
    D = np.empty(alpha.shape, dtype=np.complex128)
    small = np.abs(alpha) < 1e-12
    D[small] = N
    a = alpha[~small]
    D[~small] = np.exp(1j * a * (N - 1) / 2) * np.sin(0.5 * N * a) / np.sin(0.5 * a)
    return D


def mse_optimal_calibration_gain(f0, fs, N, L, H):
    """MSE-optimal complex calibration gain for cecHT endpoint correction.

    Ports the closed-form derivation from ECHT._calibration in
    IAF_Estimation/cecHT/phase.py so it can be precomputed here, offline,
    instead of once per IAF update at runtime in PhaseEstimator.cpp.

    Parameters
    ----------
    f0 : float
        Target (center) frequency in Hz.
    fs : float
        Sample rate in Hz.
    N : int
        Number of time-domain samples the endpoint sits at (window length,
        before FFT padding).
    L : int
        FFT length used for the bandpass frequency response `H`.
    H : ndarray, shape (L,)
        Bandpass filter frequency response on the natural (unshifted) DFT
        bin grid, i.e. H[k] corresponds to omega_k = 2*pi*k/L.
    """
    k = np.arange(L)
    omega_k = 2 * np.pi * k / L
    omega0 = 2 * np.pi * f0 / fs
    n = N - 1

    # Hilbert analytic-signal multiplier: zero above Nyquist.
    h = np.zeros(L, dtype=float)
    h[0] = 1
    if L % 2 == 0:
        h[1:L // 2] = 2
        h[L // 2] = 1
    else:
        h[1:(L + 1) // 2] = 2

    G = h * H

    X_plus = 0.5 * _dirichlet_kernel(omega0 - omega_k, N)
    X_minus = 0.5 * _dirichlet_kernel(-omega0 - omega_k, N)
    phase = np.exp(1j * omega_k * n)

    P = (G * X_plus * phase).sum() / L
    M = (G * X_minus * phase).sum() / L

    Gplus = P * np.exp(-1j * omega0 * n)
    Gminus = M * np.exp(-1j * omega0 * n)

    denom = np.abs(Gplus) ** 2 + np.abs(Gminus) ** 2
    if denom < 1e-12:
        return 1 + 0j
    return np.conj(Gplus) / denom


def gen_filter_ecHT(filter_params, output_folder):

    sos_outputs = []

    N, low_cutoff, high_cutoff, fs, length, btype, f0 = filter_params[0]

    window_length = length  # time-domain sample count (the "N" in the Dirichlet-kernel sense), before FFT padding
    if length is not None:
        # Store frequency response of bandpass filter (for PhaseEstimator)
        length = fftpack.next_fast_len(length)

    # Nudge away from exact .xx5 boundaries before rounding to 2 decimals, so that
    # tiny floating-point noise can't flip the rounding
    # direction relative to the runtime C++ computation of the same filename.
    tie_break_epsilon = 1e-9
    filename = f"{N}_{low_cutoff + tie_break_epsilon:.2f}_{high_cutoff + tie_break_epsilon:.2f}_{fs}{'_' + str(length) if length is not None else ''}.txt"
    output_path = f"{output_folder}/{filename}"
    if os.path.exists(output_path):
        return

    print(f"Generating coefficients for {low_cutoff:.2f}-{high_cutoff:.2f}Hz")
    
    if low_cutoff == 0:
        sos = butter(N=N, Wn=high_cutoff / (fs / 2), btype="low", output="sos")
    else:
        Wn = [low_cutoff / (fs / 2), high_cutoff / (fs / 2)]
        sos = butter(N=N, Wn=Wn, btype=btype, output="sos")

    sos_outputs.append(sos)

    if length is not None:
        # Store frequency response of bandpass filter (for PhaseEstimator)
        filt_freq = np.fft.fftfreq(length, d=1 / fs)
        _, H = freqz_sos(sos, worN=filt_freq, fs=fs)

        calib_gain = mse_optimal_calibration_gain(f0=f0, fs=fs, N=window_length, L=length, H=H)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("##\n")
            f.write("# type = frequency response\n")
            f.write(f"# description = Frequency response of {btype} filter {low_cutoff:.2f}-{high_cutoff:.2f}Hz @ {fs}Hz ({length} samples)\n")
            f.write(f"# calibration gain real = {calib_gain.real:.18f}\n")
            f.write(f"# calibration gain imag = {calib_gain.imag:.18f}\n")
            f.write("# format = text\n")
            f.write("##\n")

            for i in range(len(filt_freq)):
                f.write(f"{H[i].real:.18g} {H[i].imag:.18g}\n")

def gen_filter(filter_params, fs, filter_name, output_folder):

    sos_outputs = []

    filename = f"{filter_name}_{fs}.txt"
    output_path = f"{output_folder}/{filename}"

    print(f"Generating coefficients for {filter_name} filter")

    for N, low_cutoff, high_cutoff, btype in filter_params:
        print(f"Adding filter: {N}, {btype}, {low_cutoff:.2f}-{high_cutoff:.2f}Hz")

        if low_cutoff == 0:
            sos = butter(N=N, Wn=high_cutoff / (fs / 2), btype="low", output="sos")
        elif high_cutoff == 0:
            sos = butter(N=N, Wn=low_cutoff / (fs / 2), btype="high", output="sos")
        else:
            Wn = [low_cutoff / (fs / 2), high_cutoff / (fs / 2)]
            sos = butter(N=N, Wn=Wn, btype=btype, output="sos")

        sos_outputs.append(sos)

    global_filter = np.vstack(sos_outputs)

    # Store filter coefficients (for MultiChannelFilter)
    description = (f"Global filter @ {fs}Hz")

    with open(output_path, "w", encoding="utf-8") as f:
        # Header format expected by parse_file_header in lib/dsp/filter.cpp.
        f.write("##\n")
        f.write("# type = sos\n")
        f.write(f"# description = {description}\n")
        f.write("# format = text\n")
        f.write("##\n")

        # SOSFilter::FromStream expects: gain on first numeric line,
        # then flattened SOS rows as whitespace-separated values.
        f.write("1.0\n")
        for row in global_filter:
            f.write(" ".join(f"{coef:.18g}" for coef in row) + "\n")

    # Store frequency response of bandpass filter (for Phasedrift compensation in PhaseEstimator)
    freqs = np.arange(0, high_cutoff, 0.1)  # from 0 to Nyquist in 0.1 Hz steps (because IAF can change in 0.1 Hz steps)
    w, H = freqz_sos(global_filter, worN=freqs, fs=fs)
    phase = np.angle(H)  # phase shift in radians at each 0.1 Hz step

    output_path = os.path.join(output_folder, filename.replace(".txt", "_phase.txt"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("##\n")
        f.write("# type = phase shift\n")
        f.write(f"# description = Phase Shift of global filter @ {fs}Hz (0.1 Hz steps)\n")
        f.write("# format = text\n")
        f.write("##\n")

        for i in range(len(freqs)):
            f.write(f"{phase[i]:.18g}\n")

    return

def plot_filter_response(sos_by_label, fs):
    """sos_by_label: dict mapping a legend label (e.g. "N=1") to an SOS array."""
    freqs = np.arange(0, fs/2, 0.1)

    zoom_lo, zoom_hi = 7, 12
    pad_db = 0.5
    pad_deg = 0.5
    mag_zoom_min, mag_zoom_max = np.inf, -np.inf
    phase_zoom_min, phase_zoom_max = np.inf, -np.inf

    fig, ((ax1, ax1z), (ax2, ax2z)) = plt.subplots(2, 2, sharex="col", figsize=FIGSIZE, layout="constrained")

    for label, sos in sos_by_label.items():
        f, H = freqz_sos(sos, worN=freqs, fs=fs)
        magnitude_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
        phase_deg = np.angle(H, deg=True)
        zoom_mask = (f >= zoom_lo) & (f <= zoom_hi)

        ax1.plot(f, magnitude_db, linewidth=1.5, label=label)
        ax2.plot(f, phase_deg, label=label)
        ax1z.plot(f, magnitude_db, linewidth=1.5, label=label)
        ax2z.plot(f, phase_deg, label=label)

        mag_zoom_min = min(mag_zoom_min, magnitude_db[zoom_mask].min())
        mag_zoom_max = max(mag_zoom_max, magnitude_db[zoom_mask].max())
        phase_zoom_min = min(phase_zoom_min, phase_deg[zoom_mask].min())
        phase_zoom_max = max(phase_zoom_max, phase_deg[zoom_mask].max())

    ax1.set_ylabel("Amplitude [dB]")
    ax1.set_title("(a)")
    ax1.grid(True, linewidth=0.2)

    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_title("(b)")
    ax2.set_ylabel("Phase [deg]")
    ax2.grid(True, linewidth=0.3)
    ax2.set_xlim(0, 100)

    ax1z.set_title("(c)")
    ax1z.set_xlim(zoom_lo, zoom_hi)
    ax1z.set_ylim(mag_zoom_min - pad_db, mag_zoom_max + pad_db)
    ax1z.grid(True, linewidth=0.3)

    ax2z.set_title("(d)")
    ax2z.set_xlabel("Frequency [Hz]")
    ax2z.set_xlim(zoom_lo, zoom_hi)
    ax2z.set_ylim(phase_zoom_min - pad_deg, phase_zoom_max + pad_deg)
    ax2z.grid(True, linewidth=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title="Fitler order", loc="outside lower center", ncol=3, frameon=True)

    save_pgf(fig, "filter_response")
    save_png(fig, "filter_response")
    plt.show()


if __name__ == "__main__":
    # Bandpass
    fs = 10000

    sos_by_label = {}
    for N in (1, 2, 4):
        sos_bandpass = butter(
            N=N,
            Wn=[5 / (fs / 2), 16 / (fs / 2)],
            btype="bandpass",
            output="sos",
        )

        sos_bandstop = butter(
            N=N,
            Wn=[46 / (fs / 2), 54 / (fs / 2)],
            btype="bandstop",
            output="sos",
        )

        sos_by_label[f"{N}"] = np.vstack([sos_bandpass, sos_bandstop])

    plot_filter_response(sos_by_label, fs)
