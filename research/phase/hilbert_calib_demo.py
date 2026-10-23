import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
from meegkit.phase import ECHT
from scipy.fft import fft, ifft, fftshift, ifftshift, next_fast_len


# Signal parameters
FS = 10_000   # sampling rate [Hz]
F0 = 7.5      # cosine frequency [Hz]
N = 20_100   # signal_length [samples] so that cycles do not fit in evenly
L = next_fast_len(N)

# Build signal & true phase
n_idx = np.arange(N)
omega0 = 2 * np.pi * F0 / FS
signal = np.cos(omega0 * n_idx)          # pure cosine, phi0 = 0
true_phase = np.angle(np.exp(1j * omega0 * n_idx))   # wrapped to [-pi, pi]


def hilbert_calib_transform(signal, N, sfreq, f0, L):

    # Bandpass filter design for the Hilbert transform
    filt_BW = F0 / 2
    l_freq = F0 - filt_BW / 2
    h_freq = F0 + filt_BW / 2

    h, H_center = ECHT._design_bandpass(
        l_freq=l_freq, h_freq=h_freq, sfreq=FS, filt_order=1, n_fft=L, filter_type="butter",
    )
    # Apply Hilbert transform with bandpass filtering
    Xf = fft(signal, n=L)
    Xf = Xf * h
    Xf = fftshift(Xf, axes=0)
    Xf = Xf * H_center
    Xf = ifftshift(Xf, axes=0)
    hilbert_xf = ifft(Xf, n=L, axis=0)[:N]

    # Calculate calibration gains for each sample
    k = np.arange(L)
    omega0 = 2 * np.pi * f0 / sfreq
    omega_k = 2 * np.pi * k / L

    # ---- Dirichlet kernel (length-N cosine spectrum) ----
    def _dirichlet_N(alpha):
        alpha = np.asarray(alpha, dtype=float)
        D = np.empty(alpha.shape, dtype=np.complex128)
        small = np.abs(alpha) < 1e-12
        D[small] = N
        a_ns = alpha[~small]
        D[~small] = (
            np.exp(1j * a_ns * (N - 1) / 2)
            * np.sin(0.5 * N * a_ns)
            / np.sin(0.5 * a_ns)
        )
        return D

    D_plus = _dirichlet_N(omega0 - omega_k)
    D_minus = _dirichlet_N(-omega0 - omega_k)

    filt_BW = f0 / 2
    l_freq = f0 - filt_BW / 2
    h_freq = f0 + filt_BW / 2

    H_eff = ifftshift(H_center)
    G = h * H_eff

    X_plus = 0.5 * D_plus
    X_minus = 0.5 * D_minus

    calibs = np.ones(N, dtype=np.complex128)

    for n in range(N):
        # Point n of the ECHT output for each spectral component.
        phase = np.exp(1j * omega_k * n)
        P = (G * X_plus * phase).sum() / L
        M = (G * X_minus * phase).sum() / L

        # Separate contributions aligned with the ideal analytic signal.
        Gplus = P * np.exp(-1j * omega0 * n)
        Gminus = M * np.exp(-1j * omega0 * n)

        # MSE-optimal complex calibration gain and corresponding minimal error
        denom = np.abs(Gplus) ** 2 + np.abs(Gminus) ** 2
        if denom == 0:
            C_opt = 1 + 0j
        else:
            C_opt = np.conj(Gplus) / denom
        calibs[n] = C_opt

    return hilbert_xf * calibs


# Offline Hilbert (uncalibrated)
filt_BW = F0 / 2
l_freq = F0 - filt_BW / 2
h_freq = F0 + filt_BW / 2
sos = butter(1, [l_freq, h_freq], fs=FS, btype="bandpass", output="sos")
signal_filt = sosfiltfilt(sos, signal)
hilbert_filt_xf = hilbert(signal_filt, N=L)[:N]
hilbert_filt_phase = np.angle(hilbert_filt_xf)

hilbert_xf = hilbert(signal, N=L)[:N]
hilbert_phase = np.angle(hilbert_xf)

# Offline Hilbert with calibration
hilbert_calib_xf = hilbert_calib_transform(signal, N, FS, F0, L)
hilbert_calib_phase = np.angle(hilbert_calib_xf)

# Phase errors (wrapped to [-180, 180] deg)
def phase_error_deg(estimated, reference):
    return np.degrees(np.angle(np.exp(1j * (reference - estimated))))


err_uncalib = phase_error_deg(hilbert_phase, true_phase)
err_filt_uncalib = phase_error_deg(hilbert_filt_phase, true_phase)
err_calib = phase_error_deg(hilbert_calib_phase, true_phase)


# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True,
                         sharex=True)

# Phase estimates
ax = axes[0]
ax.plot(n_idx, true_phase,          lw=1.5,
        label="True phase",             zorder=3)
ax.plot(n_idx, hilbert_phase,       lw=1.2, alpha=0.8, ls="--", c="tab:orange",
        label="Hilbert (uncalibrated)")
ax.plot(n_idx, hilbert_filt_phase, lw=1.2, alpha=0.8, ls=":", c="tab:red",
        label="Hilbert (filtered, uncalibrated)")
ax.plot(n_idx, hilbert_calib_phase, lw=1.2, alpha=0.8, ls="-.", c="tab:green",
        label="Hilbert (calibrated)")
ax.set_ylabel("Phase (rad)")
ax.set_title(f"Phase estimates  —  f0={F0} Hz, Fs={FS} Hz, N={N} samples")
ax.legend(frameon=False, ncol=3)
ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.7)

# Phase errors
ax = axes[1]
ax.plot(n_idx, err_uncalib, lw=1.0, alpha=0.8, label="Uncalibrated error", ls="--", c="tab:orange")
ax.plot(n_idx, err_filt_uncalib, lw=1.0, alpha=0.8, label="Filtered uncalibrated error", ls=":", c="tab:red")
ax.plot(n_idx, err_calib,   lw=1.0, alpha=0.8, ls="-.", label="Calibrated error", c="tab:green")
ax.set_ylabel("Phase error (deg)")
ax.set_title("Phase error vs true phase")
ax.legend(frameon=False, ncol=2)
ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.7)

# ax.set_ylim(-2, 2)

fig.savefig("hilbert_calib_demo.png", dpi=150, bbox_inches="tight")
plt.show()

# Summary statistics
for label, err in [("Uncalibrated", err_uncalib), ("Filtered uncalibrated", err_filt_uncalib), ("Calibrated", err_calib)]:
    print(f"\n{label}:")
    print(f"mean={np.nanmean(err):+.3f}°  "
          f"std={np.nanstd(err):.3f}°  max|err|={np.max(np.abs(err)):.3f}°")
