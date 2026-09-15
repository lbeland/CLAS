"""
Test script: does spectral whitening (removal of the 1/f aperiodic component)
improve or degrade instantaneous-phase estimation of an alpha oscillation,
compared to plain bandpass + Hilbert transform?

Signal model
------------
`x = pink_noise + tone`, where `pink_noise` has PSD(f) = 10**offset * f**-exponent
(each frequency bin gets amplitude sqrt(PSD) and an independent uniform
random phase, then `irfft`) and `tone` is a single sinusoid at `tone_freq`
with a random start phase. The ground-truth instantaneous phase is just the
tone's own analytic phase, known exactly in closed form.

Four pipelines are compared (all zero-phase / offline, for a fair comparison
with FFT-whitening, which is inherently zero-phase):

  1. bandpass          : hilbert(bandpass(x))
  2. whiten+bandpass    : hilbert(bandpass(whiten(x)))
  3. extendedHT         : phase_reconst(bandpass(x))              (see extended_HT.py)
  4. bandpass+JADE      : JADE(bandpass(x))                       (see jade.py)

Whitening uses the *exact* known aperiodic (1/f) parameters, so any phase
error it introduces is attributable to the whitening operation itself (noise
amplification at high frequencies, spectral leakage / edge effects), not to
fitting inaccuracy.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.signal.windows import tukey
import matplotlib as mpl

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import extended_HT as ht
from jade import jade_v3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.plot import _circ_stats, _style_polar_axis, COMMON_BBOX

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config = {
    "fs": 10000,
    "signal_length_sec": 10.0,

    # Pink-noise background: PSD(f) = 10**offset * f**-exponent
    "aperiodic_params": [1.0, 2.0],   # [offset, exponent]

    # Single-tone oscillation
    "tone_freq": 11.0,   # Hz
    "tone_amp": 1.0,     # signal amplitude (not PSD)

    # Processing
    "band": (6.0, 15.0),                # bandpass range
    "edge_margin_sec": 2.0,             # excluded from error stats (not from plots)
}

fs = config["fs"]
n = int(config["signal_length_sec"] * fs)
t = np.arange(n) / fs

# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def aperiodic_psd(freqs, aperiodic_params):
    """PSD(f) = 10**offset * f**-exponent. Undefined at f=0; caller must mask."""
    offset, exponent = aperiodic_params
    return 10 ** offset * freqs ** -exponent


def generate_pink_noise(n_samples, fs, aperiodic_params, rng):
    """Synthesize real pink noise with PSD(f) = aperiodic_psd(f), by assigning
    each frequency bin amplitude sqrt(power) and an independent random phase,
    then `irfft`."""
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
    amp = np.zeros_like(freqs)
    # /2 compensates for the one-sided-PSD doubling applied to AC bins when
    # measuring PSD back from a signal (e.g. psd_raw[1:-1] *= 2 below), so
    # the realized PSD of the output matches aperiodic_psd exactly, not 2x it.
    amp[1:] = np.sqrt(aperiodic_psd(freqs[1:], aperiodic_params) * fs * n_samples / 2)
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    X = amp * np.exp(1j * phases)
    return np.fft.irfft(X, n=n_samples)


def generate_signal_with_ground_truth_phase(n_samples, fs, aperiodic_params, tone_freq, tone_amp, rng):
    """x = pink_noise + tone. The ground-truth instantaneous phase is just
    the tone's own analytic phase, known exactly in closed form."""
    pink = generate_pink_noise(n_samples, fs, aperiodic_params, rng)

    t = np.arange(n_samples) / fs
    tone_phase0 = rng.uniform(0, 2 * np.pi)
    true_phase = np.angle(np.exp(1j * (2 * np.pi * tone_freq * t + tone_phase0)))
    tone = tone_amp * np.cos(true_phase)

    return pink + tone, true_phase


ap_params = config["aperiodic_params"]
center_freq = config["tone_freq"]
tone_amp = config["tone_amp"]

ap_psd_at_cf = aperiodic_psd(np.array([center_freq]), ap_params)[0]
print(f"Aperiodic PSD @ {center_freq} Hz: {10 * np.log10(ap_psd_at_cf):.2f} dB")
print(f"Tone amplitude @ {center_freq} Hz: {tone_amp}")

x, true_phase = generate_signal_with_ground_truth_phase(
    n, fs, ap_params, center_freq, tone_amp, rng
)

# ---------------------------------------------------------------------------
# Processing pipelines
# ---------------------------------------------------------------------------

def bandpass_filter(sig, fs, band, order=4):
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def whiten(sig, fs, aperiodic_params):
    """Zero-phase whitening using the EXACT known aperiodic model
    (aperiodic_psd, same parametrization used for signal generation)."""
    n_samples = len(sig)
    sig = sig * tukey(n_samples, alpha=0.05)  # taper edges to reduce spectral leakage
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
    X = np.fft.rfft(sig)

    L = np.empty_like(freqs)
    L[1:] = aperiodic_psd(freqs[1:], aperiodic_params)
    L[0] = L[1]

    X_white = X / np.sqrt(L)  # magnitude scaled, phase untouched (L real & positive)
    return np.fft.irfft(X_white, n=n_samples)


band = config["band"]
sig_bp = bandpass_filter(x, fs, band)
sig_white = whiten(x, fs, ap_params)
sig_white_bp = bandpass_filter(sig_white, fs, band)

pipelines = {
    "1. bandpass":        sig_bp,
    "2. whiten+bandpass": sig_white_bp,
}

# ---------------------------------------------------------------------------
# Phase estimation + circular error
# ---------------------------------------------------------------------------

def circular_error(est_phase, true_phase):
    """Wrapped phase difference, result in (-pi, pi]."""
    return np.angle(np.exp(1j * (est_phase - true_phase)))


edge_n = int(config["edge_margin_sec"] * fs)
valid = slice(edge_n, n - edge_n)

results = {}
for name, sig in pipelines.items():
    est_phase = np.angle(hilbert(sig))
    err = circular_error(est_phase, true_phase)
    mu, sd, _, _ = _circ_stats(err[valid])
    results[name] = {
        "phase": est_phase,
        "error": err,
        "circ_mean_deg": np.degrees(mu),
        "circ_std_deg": np.degrees(sd),
        "mae_deg": np.degrees(np.mean(np.abs(err[valid]))),
        "max_abs_deg": np.degrees(np.max(np.abs(err[valid]))),
    }
# # phase_reconst assumes the input wraps around smoothly (it takes a single
# # whole-signal FFT internally); sig_bp's raw start/end values don't match
# # (~0.5x the signal's own std), so taper the edges toward zero first. Keep
# # the tapered region inside edge_margin_sec, which is already excluded from
# # the error stats, so it doesn't affect the scored samples.
# taper_alpha = min(1.0, 2 * config["edge_margin_sec"] / config["signal_length_sec"])
# sig_bp_tapered = sig_bp * tukey(len(sig_bp), alpha=taper_alpha)
# phase_extHT_cut, _, extHT_initindx = ht.phase_reconst(sig_bp_tapered.reshape(1, -1), 1/fs)
# phase_extHT = np.full(n, np.nan)
# phase_extHT[extHT_initindx:extHT_initindx + phase_extHT_cut.shape[1]] = phase_extHT_cut[0]
# err = circular_error(phase_extHT, true_phase)
# mu, sd, _, _ = _circ_stats(circular_error(phase_extHT, true_phase)[valid])
# results["3. extendedHT"] = {
#     "phase": phase_extHT,
#     "error": err,
#     "circ_mean_deg": np.degrees(mu),
#     "circ_std_deg": np.degrees(sd),
#     "mae_deg": np.degrees(np.mean(np.abs(err[valid]))),
#     "max_abs_deg": np.degrees(np.max(np.abs(err[valid]))),
# }

# # JADE (DTW-based phase/frequency estimation, see jade.py) applied directly
# # to the bandpassed signal. JADE only produces phase samples from its first
# # to its last detected zero crossing (near the very start/end of sig_bp), so
# # fill the rest with NaN the same way phase_extHT does above.
# #
# # `phasevals` is a continuous phase in *cycles*, anchored to JADE's own
# # zero-crossing convention: x(t) ~ sign * amplitude(t) * sin(2*pi*phasevals),
# # phase 0 at a zero crossing, where `sign` (+-1, constant) depends on whether
# # the very first detected zero crossing is rising or falling. `true_phase`
# # (and the Hilbert-based pipelines) instead use the analytic-signal
# # convention x(t) ~ amplitude(t) * cos(Phi(t)), phase 0 at a peak. Since
# # sin(a) = cos(a - pi/2), converting with plain 2*pi*phasevals leaves a
# # constant +-90 deg quadrature offset relative to the other pipelines.
# # Convert properly (Phi = 2*pi*phasevals - sign*pi/2), determining `sign`
# # straight from JADE's own output (correlate its zero-crossing-convention
# # reconstruction against the actual bandpassed signal) rather than assuming
# # a fixed sign, since it flips depending on the data.
# jade_IFvals, jade_phasevals, jade_zc, jade_amplitudegram = jade_v3(
#     sig_bp, t, 1 / fs, smooth=0, normalize=0)
# jade_start = int(jade_zc[0]) - 1
# jade_recon = jade_amplitudegram * np.sin(2 * np.pi * jade_phasevals)
# jade_sign = 1.0 if np.dot(jade_recon, sig_bp[jade_start:jade_start + jade_recon.size]) >= 0 else -1.0
# phase_jade = np.full(n, np.nan)
# phase_jade[jade_start:jade_start + jade_phasevals.size] = (
#     2 * np.pi * jade_phasevals - jade_sign * np.pi / 2)
# err = circular_error(phase_jade, true_phase)
# mu, sd, _, _ = _circ_stats(err[valid])
# results["4. bandpass+JADE"] = {
#     "phase": phase_jade,
#     "error": err,
#     "circ_mean_deg": np.degrees(mu),
#     "circ_std_deg": np.degrees(sd),
#     "mae_deg": np.degrees(np.mean(np.abs(err[valid]))),
#     "max_abs_deg": np.degrees(np.max(np.abs(err[valid]))),
# }

print(f"\n{'Pipeline':<22} {'circ. mean (deg)':>18} {'circ. std (deg)':>18} {'MAE (deg)':>12} {'max |err| (deg)':>18}")
print("-" * 90)
for name, r in results.items():
    print(f"{name:<22} {r['circ_mean_deg']:>18.3f} {r['circ_std_deg']:>18.3f} {r['mae_deg']:>12.3f} {r['max_abs_deg']:>18.3f}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
plt.figure()
plt.plot(t,x)
plt.xlabel("Time (s)")
plt.show()

n_results = len(results)
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, n_results)
axes = [
    fig.add_subplot(gs[0, :]),
]
ax_polars = [fig.add_subplot(gs[1, i], projection="polar") for i in range(n_results)]

# --- PSD sanity check ---
ax = axes[0]
freqs_rfft = np.fft.rfftfreq(n, d=1 / fs)
psd_raw = (np.abs(np.fft.rfft(x)) ** 2) / (fs * n)
psd_raw[1:-1] *= 2
psd_white = (np.abs(np.fft.rfft(sig_white)) ** 2) / (fs * n)
psd_white[1:-1] *= 2
psd_bp = (np.abs(np.fft.rfft(sig_bp)) ** 2) / (fs * n)
psd_bp[1:-1] *= 2
psd_white_bp = (np.abs(np.fft.rfft(sig_white_bp)) ** 2) / (fs * n)
psd_white_bp[1:-1] *= 2

freqs_nz = freqs_rfft[1:]
ap_only = aperiodic_psd(freqs_nz, ap_params)

mask = (freqs_rfft <= 40) & (freqs_rfft > 0)
mask_nz = freqs_nz <= 40
ax.semilogy(freqs_rfft[mask],psd_raw[mask], label="raw PSD", alpha=0.7)
ax.semilogy(freqs_nz[mask_nz], ap_only[mask_nz], "k--", label=r"aperiodic model $L(f)$")
ax.semilogy(freqs_rfft[mask], psd_white[mask], label="whitened PSD", alpha=0.7)
ax.semilogy(freqs_rfft[mask], psd_bp[mask], label="bandpass PSD", alpha=0.7)
ax.semilogy(freqs_rfft[mask], psd_white_bp[mask], label="whitened+bandpass PSD", alpha=0.7)
ax.axvline(center_freq, color="g", alpha=0.4, label=f"f0={center_freq} Hz")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.legend(fontsize=8)

# --- Circular phase error distributions, one polar plot per method ---
# (same style as analysis/plot.py's plot_errors())
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
bin_width_deg = 10
bin_width_rad = np.radians(bin_width_deg)
bins = np.arange(-180, 181, bin_width_deg)
centers = (np.radians(bins[:-1]) + np.radians(bins[1:])) / 2

# Same `valid` window used for the printed table, so the numbers agree.
names = list(results.keys())
hists = []
for name in names:
    vals = np.degrees(results[name]["error"][valid])
    hists.append(np.histogram(vals, bins=bins)[0] / max(1, vals.size) * 100)

r_max = max(h.max() for h in hists) * 1.05
r_max = max(r_max, 5)
r_ticks = [rt for rt in [10, 20, 30] if rt < r_max]

for i, name in enumerate(names):
    color = colors[i % len(colors)]
    mu_u = np.radians(results[name]["circ_mean_deg"])
    sd_u = np.radians(results[name]["circ_std_deg"])

    ax_polars[i].bar(centers, hists[i], width=bin_width_rad,
                      color=color, edgecolor="0", linewidth=0.75)
    ax_polars[i].plot([mu_u, mu_u], [0, r_max], color="0", linewidth=2)
    _style_polar_axis(ax_polars[i], r_max, r_ticks)
    ax_polars[i].text(
        0.5, 0.45,
        rf"${np.round(np.degrees(mu_u), 1) + 0.0:.1f}^\circ"
        rf" \pm {np.round(np.degrees(sd_u), 1):.1f}^\circ$",
        transform=ax_polars[i].transAxes, bbox=COMMON_BBOX, va="top", ha="center",
    )
    ax_polars[i].set_title(name, fontsize=10)

fig.suptitle(
    f"fs={fs} Hz, offset={ap_params[0]}, exponent={ap_params[1]}, f0={center_freq} Hz, "
    f"tone_amp={tone_amp}, band={band}",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()
plt.savefig("phase_test.png", dpi=300)
