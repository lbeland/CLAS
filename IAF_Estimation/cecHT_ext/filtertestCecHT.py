"""
Compare the default causal ECHT (Butterworth band-pass) against a
zero-phase variant that uses the same Butterworth design forward-backward
(|H(f)|**2), using a synthetic sine wave with known ground-truth phase.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
import _bootstrap  # noqa: E402,F401  -> puts the cecHT submodule on sys.path

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from echt_ext import ECHTExt as ECHT  # noqa: E402


def wrap_phase(phi):
    """Wrap phase angle(s) to [-pi, pi]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


class ZeroPhaseButterECHT(ECHT):
    """ECHT variant emulating a forward-backward (filtfilt-style) zero-phase
    Butterworth band-pass.

    Applying an IIR filter forward and then backward multiplies the spectrum
    by H(f) and then by conj(H(f)), i.e. by |H(f)|**2 -- a real, non-negative
    frequency response whose phase is identically zero. Here we take the exact
    same Butterworth design the default ECHT uses and substitute |H(f)|**2 for
    H(f) as the band-pass response, so within the ECHT's own frequency-domain,
    sliding-window machinery the band-pass contributes amplitude shaping only
    and introduces no phase distortion."""

    def _design_bandpass(self, l_freq, h_freq, sfreq, filt_order, n_fft, filter_type="butter", b=None, a=None):
        h, H_center = super()._design_bandpass(
            l_freq, h_freq, sfreq, filt_order, n_fft, filter_type="butter",
        )
        # |H(f)|**2: forward * backward pass -> zero-phase magnitude response
        return h, np.abs(H_center) ** 2


# ---------------------------------------------------------------------
# Signal: 10 Hz sine, 20 s @ 10 kHz, with known ground-truth phase
# ---------------------------------------------------------------------
f0 = 9
fs = 10000.0
duration = 3.0
l_freq, h_freq = 5.0, 16.0

t = np.arange(0, duration, 1 / fs)
# Use a cosine, not a sine: the analytic signal (Hilbert transform) of
# cos(theta) has phase exactly theta, so the ground-truth phase below
# matches the ECHT/analytic-signal convention directly. A sine would carry
# a constant -90 deg offset relative to its own analytic phase, which would
# swamp the (much smaller) filter-induced errors we actually want to compare.
x = np.cos(2 * np.pi * f0 * t) + 0.1 * np.random.randn(len(t))
phase_gt = wrap_phase(2 * np.pi * f0 * t)

# ---------------------------------------------------------------------
# Sliding-window ECHT: slide a 2000-sample window through the signal one
# sample at a time, ECHT-transform each window, and keep only its endpoint
# (the causal, "real-time" estimate). The first window_size-1 samples have
# no full window yet, so they're left as NaN (ignored).
# ---------------------------------------------------------------------
window_size = 2000
n_samples = len(x)


def run_windowed(*echts):
    phases = [np.full(n_samples, np.nan) for _ in echts]
    for echt in echts:
        echt.fit(x[:window_size])  # fixes n_fft/h_/coef_/calibration once, reused every window
    for end in range(window_size - 1, n_samples):
        window = x[end - window_size + 1: end + 1]
        for echt, phase in zip(echts, phases):
            phase[end] = np.angle(echt.transform(window)[-1, 0])
    return phases


# ---------------------------------------------------------------------
# Default ECHT (Butterworth band-pass, 9-11 Hz)
# ---------------------------------------------------------------------
echt_default = ECHT(
    l_freq=l_freq, h_freq=h_freq, sfreq=fs, filter_type="butter",
    n_fft=window_size, fft_mode="exact", f0=f0, calibrate=True,
)

# ---------------------------------------------------------------------
# Zero-phase ECHT: same Butterworth band-pass as echt_default, but used
# forward-backward (|H(f)|**2) so it contributes no phase distortion. Run
# through the identical sliding-window machinery as the default variant.
# calibrate=True works here because ECHT._calibration now goes through
# self._design_bandpass, so the calibration gain is fitted to this
# variant's own |H(f)|**2 response (finite-window endpoint amplitude
# correction) rather than to the plain causal Butterworth response.
# ---------------------------------------------------------------------
echt_zerophase = ZeroPhaseButterECHT(
    l_freq=l_freq, h_freq=h_freq, sfreq=fs, filter_type="butter",
    n_fft=window_size, fft_mode="exact", f0=f0, calibrate=True,
)

# Run the sliding-window ECHT for both variants
phase_default, phase_zerophase = run_windowed(echt_default, echt_zerophase)

# ---------------------------------------------------------------------
# Phase error vs. ground truth (only where a full window is available)
# ---------------------------------------------------------------------
valid = slice(window_size - 1, None)
err_default = np.degrees(wrap_phase(phase_default[valid] - phase_gt[valid]))
err_zerophase = np.degrees(wrap_phase(phase_zerophase[valid] - phase_gt[valid]))


def summarize(label, err):
    print(
        f"{label}: mean|err| = {np.abs(err).mean():.3f} deg, "
        f"std = {err.std(ddof=0):.3f} deg, max|err| = {np.abs(err).max():.3f} deg"
    )


summarize("Default ECHT (Butterworth 9-11 Hz)", err_default)
summarize("Zero-phase ECHT (Butterworth fwd-bwd)", err_zerophase)

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

axes[0].plot(t, phase_gt, label="Ground truth", lw=1.5)
axes[0].plot(t, phase_default, label="Default (Butterworth)", alpha=0.7)
axes[0].plot(t, phase_zerophase, label="Zero-phase (Butterworth fwd-bwd)", alpha=0.7)
axes[0].set_ylabel("Phase (rad)")
axes[0].legend(loc="upper right")

axes[1].plot(t[valid], err_default, label="Default (Butterworth)")
axes[1].plot(t[valid], err_zerophase, label="Zero-phase (Butterworth fwd-bwd)")
axes[1].set_ylabel("Phase error (deg)")
axes[1].set_xlabel("Time (s)")
axes[1].legend(loc="upper right")

plt.tight_layout()
_OUT = pathlib.Path(__file__).resolve().parent / "figures" / "filtertestCecHT.png"
plt.savefig(_OUT, dpi=300)
plt.show()
