"""
Test script: does spectral whitening (removal of the 1/f aperiodic component)
improve or degrade instantaneous-phase estimation of an alpha oscillation,
compared to plain bandpass + Hilbert transform?

Signal model (log-space, FOOOF generative model)
--------------------------------------------------
Reuses the aperiodic / periodic / noise generators from
`fooof.sim.gen` (also used by IAF_Estimation/SIMparam/code/sims.py) so the
simulated power spectrum follows the same model FOOOF fits:

    log10(power(f)) = aperiodic(f) + gaussian_peak(f) + noise(f)

  - aperiodic(f)      : gen_aperiodic(f, [offset, exponent])   -> offset - exponent*log10(f)
  - gaussian_peak(f)  : gen_periodic(f, [cen, height, bw])     -> Gaussian bump in log10-power
  - noise(f)          : gen_noise(f, nlv)                      -> N(0, nlv) per bin, in log10-power

`sims.gen_power_vals_fn` combines these into a single linear power spectrum
`10**(aperiodic + peak + noise)`. A real time-domain signal is then
synthesized by treating that spectrum as a target one-sided amplitude
spectrum: each frequency bin gets amplitude sqrt(power) and an independent
uniform random phase, followed by `irfft`.

Ground-truth instantaneous phase
---------------------------------
The oscillation is embedded in `x` as a Gaussian *bump on top of* the
aperiodic background (not as an independent additive component), so there is
no analytic signal for "the peak alone" embedded in `x` directly. Instead,
the peak's own spectral content (gen_periodic alone, no aperiodic/noise) is
reconstructed using the same per-bin random phases used to build `x`, and
its exact analytic signal is taken directly (no Hilbert call): keep DC
as-is, double the other positive-frequency bins, zero out negative
frequencies. A pure-sinusoid-at-center_freq alternative (ignoring the
Gaussian spread of power around center_freq entirely) is kept commented out
in `generate_signal_with_ground_truth_phase` for comparison.

Four pipelines are compared (all zero-phase / offline, for a fair comparison
with FFT-whitening, which is inherently zero-phase):

  1. bandpass          : hilbert(bandpass(x))
  2. whiten             : hilbert(whiten(x))                    (no bandpass!)
  3. whiten+bandpass    : hilbert(bandpass(whiten(x)))

Whitening uses the *exact* known aperiodic parameters (no fitting, via
`gen_aperiodic`), so any phase error it introduces is attributable to the
whitening operation itself (noise amplification at high frequencies,
spectral leakage / edge effects), not to fitting inaccuracy.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "IAF_Estimation" / "SIMparam" / "code"))
from fooof.sim.gen import gen_aperiodic, gen_periodic, gen_noise  # noqa: E402
from sims import gen_power_vals_fn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.plot import _circ_stats, _style_polar_axis, COMMON_BBOX  # noqa: E402

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config = {
    "fs": 10000,
    "signal_length_sec": 30.0,

    # Aperiodic background: gen_aperiodic([offset, exponent]) -> log10 power
    "aperiodic_params": [1.0, 2.0],

    # Periodic peak: gen_periodic([center_freq, height, bw]) -> log10 power
    # `height` is how many log10-power units (dB/10) the peak sits above the
    # aperiodic component AT its own center frequency.
    "peak_params": [7.0, 2.0, 0.1],

    # Noise added in log10-power space, per frequency bin (gen_noise).
    "noise_lv": 0.1,

    # Processing
    "band": (6.0, 16.0),                # bandpass range
    "edge_margin_sec": 2.0,             # excluded from error stats (not from plots)
}

fs = config["fs"]
n = int(config["signal_length_sec"] * fs)
t = np.arange(n) / fs

# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signal_with_ground_truth_phase(n_samples, fs, aperiodic_params, peak_params, nlv, rng):
    """
    Synthesize a real signal whose power spectrum follows the FOOOF
    generative model (log10 power = aperiodic + gaussian peak + noise, via
    fooof.sim.gen / sims.gen_power_vals_fn), by assigning each frequency bin
    amplitude sqrt(power) and an independent random phase, then `irfft`.

    Also returns the ground-truth instantaneous phase of the embedded
    oscillation, taken from the exact analytic signal of the isolated
    Gaussian peak content (gen_periodic alone, same per-bin random phases as
    the full signal). A pure-sinusoid-at-center_freq alternative is kept
    commented out below it for comparison.
    """
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
    center_freq = peak_params[0]

    # gen_aperiodic/gen_periodic are undefined at f=0 (log10(0)); model the
    # spectrum over the non-DC bins only and leave the DC bin at zero.
    freqs_nz = freqs[1:]
    powers_nz = gen_power_vals_fn(
        freqs_nz,
        ap_kwargs={"aperiodic_params": aperiodic_params},
        pe_kwargs={"periodic_params": peak_params},
        noise_kwargs={"nlv": nlv},
        ap_func=gen_aperiodic, pe_func=gen_periodic, noise_func=gen_noise,
    )

    amp = np.zeros_like(freqs)
    # /2 compensates for the one-sided-PSD doubling applied to AC bins when
    # measuring PSD back from a signal (e.g. psd_raw[1:-1] *= 2 below), so
    # the realized PSD of `sig` matches `powers_nz` exactly, not 2x it.
    amp[1:] = np.sqrt(powers_nz * fs * n_samples / 2)

    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    X = amp * np.exp(1j * phases)
    sig = np.fft.irfft(X, n=n_samples)

    # --- Ground truth as pure sinusoid at center_freq (ignores the Gaussian
    # spread of power around center_freq) ---
    # center_bin = np.argmin(np.abs(freqs - center_freq))
    # phase0 = phases[center_bin]
    # t = np.arange(n_samples) / fs
    # true_phase = np.angle(np.exp(1j * (2 * np.pi * center_freq * t + phase0)))

    # --- Ground truth from the whole Gaussian peak ---
    # Isolate the peak's own spectral content, reusing the SAME per-bin
    # random phases assigned above, and build its exact analytic signal
    # directly (no Hilbert call): keep DC as-is, double the other
    # positive-frequency bins, zero out negative frequencies.
    # gen_periodic is an additive log10-power term that -> 0 (not -inf) far
    # from the peak (it means "no change to the aperiodic level" there), so
    # 10**gen_periodic -> 1, not 0. Subtract that floor off so the isolated
    # peak power genuinely -> 0 away from center_freq.
    peak_power_nz = 10 ** gen_periodic(freqs_nz, peak_params) - 1
    peak_amp = np.zeros_like(freqs)
    peak_amp[1:] = np.sqrt(peak_power_nz * fs * n_samples)
    X_peak = peak_amp * np.exp(1j * phases)

    n_bins = len(X_peak)  # == n_samples // 2 + 1
    X_full = np.zeros(n_samples, dtype=complex)
    X_full[0] = X_peak[0]
    if n_samples % 2 == 0:
        X_full[1:n_bins - 1] = 2 * X_peak[1:n_bins - 1]
        X_full[n_bins - 1] = X_peak[n_bins - 1]  # Nyquist (n_samples even)
    else:
        X_full[1:n_bins] = 2 * X_peak[1:n_bins]

    analytic = np.fft.ifft(X_full)
    true_phase = np.angle(analytic)

    return sig, true_phase, freqs, powers_nz


ap_params = config["aperiodic_params"]
peak_params = config["peak_params"]
center_freq = peak_params[0]

ap_log_at_cf = gen_aperiodic(np.array([center_freq]), ap_params)[0]
print(f"Aperiodic PSD @ {center_freq} Hz: {10 * ap_log_at_cf:.2f} dB")
print(f"Peak height @ {center_freq} Hz: {10 * peak_params[1]:+.1f} dB above aperiodic "
      f"-> target peak PSD: {10 * (ap_log_at_cf + peak_params[1]):.2f} dB")

x, true_phase, freqs_rfft, powers_nz = generate_signal_with_ground_truth_phase(
    n, fs, ap_params, peak_params, config["noise_lv"], rng
)

# ---------------------------------------------------------------------------
# Processing pipelines
# ---------------------------------------------------------------------------

def bandpass_filter(sig, fs, band, order=4):
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def whiten(sig, fs, aperiodic_params):
    """Zero-phase whitening using the EXACT known aperiodic model
    (gen_aperiodic, same parametrization used for signal generation)."""
    n_samples = len(sig)
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
    X = np.fft.rfft(sig)

    L_log = np.empty_like(freqs)
    L_log[1:] = gen_aperiodic(freqs[1:], aperiodic_params)
    L_log[0] = L_log[1]
    L = 10 ** L_log

    X_white = X / np.sqrt(L) - 1  # magnitude scaled, phase untouched (L real & positive)
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

print(f"\n{'Pipeline':<22} {'circ. mean (deg)':>18} {'circ. std (deg)':>18} {'MAE (deg)':>12} {'max |err| (deg)':>18}")
print("-" * 90)
for name, r in results.items():
    print(f"{name:<22} {r['circ_mean_deg']:>18.3f} {r['circ_std_deg']:>18.3f} {r['mae_deg']:>12.3f} {r['max_abs_deg']:>18.3f}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
plt.figure()
plt.plot(x)
plt.show()

n_pipelines = len(pipelines)
fig = plt.figure(figsize=(12, 14))
gs = fig.add_gridspec(3, n_pipelines)
axes = [
    fig.add_subplot(gs[0, :]),
    fig.add_subplot(gs[1, :]),
    None,
]
ax_polars = [fig.add_subplot(gs[2, i], projection="polar") for i in range(n_pipelines)]

# --- PSD sanity check ---
ax = axes[0]
psd_raw = (np.abs(np.fft.rfft(x)) ** 2) / (fs * n)
psd_raw[1:-1] *= 2
psd_white = (np.abs(np.fft.rfft(sig_white)) ** 2) / (fs * n)
psd_white[1:-1] *= 2
psd_bp = (np.abs(np.fft.rfft(sig_bp)) ** 2) / (fs * n)
psd_bp[1:-1] *= 2
psd_white_bp = (np.abs(np.fft.rfft(sig_white_bp)) ** 2) / (fs * n)
psd_white_bp[1:-1] *= 2

freqs_nz = freqs_rfft[1:]
ap_only = 10 ** gen_aperiodic(freqs_nz, ap_params)
ap_plus_peak = 10 ** (gen_aperiodic(freqs_nz, ap_params) + gen_periodic(freqs_nz, peak_params))

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

# --- Circular phase error over time ---
ax = axes[1]
for name, r in results.items():
    ax.plot(t, np.degrees(r["error"]), lw=0.6, alpha=0.7, label=name)
ax.axvline(config["edge_margin_sec"], color="gray", ls=":", lw=1)
ax.axvline(config["signal_length_sec"] - config["edge_margin_sec"], color="gray", ls=":", lw=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Phase error (deg)")
ax.set_title("Circular phase error over time")
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
    f"bw={peak_params[2]} Hz, band={band}, peak={10 * peak_params[1]:+.0f} dB above aperiodic @ f0",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()
plt.savefig("phase_test.png", dpi=300)
