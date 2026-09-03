"""Example/test script for jade.py driven by a real FIF decomposition.

Builds a synthetic multi-component signal with known ground-truth
instantaneous frequencies (a frequency-modulated "alpha" tone, an
amplitude-modulated "beta" tone and a slow trend), decomposes it into
Intrinsic Mode Components with Fast Iterative Filtering (FIF), then runs
JADE on the individual IMCs and compares the reconstruction and the
estimated instantaneous frequency against the analytic ground truth.

This mirrors the structure of JADE_example.m (nonlinear oscillator + FIF +
per-IMF JADE), but replaces the Duffing/Euler oscillator with a closed-form
test signal so the estimates can be checked against an exact reference.

Requires the bundled FIF package (PHASE_estimation/FIF, needs numpy, scipy,
numba).
"""

import os
import sys
import contextlib

import numpy as np
import matplotlib.pyplot as plt

# FIF lives in a sub-package next to this script; make it importable no matter
# what the current working directory is.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import FIF  # noqa: E402
from jade import jade_v3  # noqa: E402


@contextlib.contextmanager
def _quiet():
    """Silence FIF's very chatty per-IMF stdout prints."""
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


# --- Build a synthetic multi-component test signal ----------------------------
Fs = 1000.0        # Hz
Ts = 1.0 / Fs
T = 10.0           # seconds
t = np.arange(0, T, Ts)

# Component 1: frequency-modulated "alpha" tone.
#   f1(t) = f0_1 + df_1 * sin(2*pi*fm_1*t)
f0_1, df_1, fm_1 = 10.0, 2.0, 0.5
true_IF_1 = f0_1 + df_1 * np.sin(2 * np.pi * fm_1 * t)
true_phase_1 = 2 * np.pi * (
    f0_1 * t - (df_1 / (2 * np.pi * fm_1)) * (np.cos(2 * np.pi * fm_1 * t) - 1)
)
comp_1 = np.sin(true_phase_1)

# Component 2: amplitude-modulated "beta" tone at a fixed carrier.
f0_2 = 30.0
true_IF_2 = np.full_like(t, f0_2)
amp_2 = 1.0 + 0.4 * np.sin(2 * np.pi * 0.3 * t)
comp_2 = amp_2 * np.sin(2 * np.pi * f0_2 * t)

# Slow trend that FIF should peel off into the last IMC.
trend = 0.5 * np.sin(2 * np.pi * 0.15 * t)

x = comp_1 + comp_2 + trend

components = [
    dict(name='FM alpha tone', true_IF=true_IF_1, true_sig=comp_1),
    dict(name='AM beta tone', true_IF=true_IF_2, true_sig=comp_2),
]

# --- FIF decomposition --------------------------------------------------------
# alpha (mask-length percentile) and ExtPoints (minimum # of extrema before the
# decomposition stops) are tuned so the two tones land in their own IMCs.
fif = FIF.FIF(alpha=90, ExtPoints=30)
with _quiet():
    fif.run(x)
    imc_freqs, _ = fif.get_freq_amplitudes(dt=Ts, as_output=True)

IMC = fif.data['IMC']          # shape (n_imc, N); last row is the trend
n_imc = IMC.shape[0]
print(f"FIF produced {n_imc} IMCs; mean frequencies (Hz): "
      f"{np.array2string(imc_freqs, precision=2)}")


# --- Per-component JADE analysis --------------------------------------------
def analyze(sig):
    """Run JADE on one IMC and return aligned slices for plotting."""
    IFvals, phasevals, zero_crossings, amplitudegram = jade_v3(
        sig, t, Ts, smooth=0, normalize=0)

    fz, lz = int(zero_crossings[0]), int(zero_crossings[-1])

    # phasevals accumulates monotonically from 0 regardless of whether the first
    # zero crossing is rising or falling, so the reconstruction can come out
    # phase-inverted. Fix the sign empirically against the IMC itself.
    reconstr = amplitudegram * np.sin(2 * np.pi * phasevals)

    n = min(lz - fz, reconstr.size, IFvals.size)
    sl = slice(fz - 1, fz - 1 + n)
    if np.dot(reconstr[:n], sig[sl]) < 0:
        reconstr = -reconstr

    return dict(fz=fz, n=n, sl=sl,
                t=t[sl], orig=sig[sl],
                reconstr=reconstr[:n],
                est_IF=IFvals[:n])  # IFvals is already in Hz


for comp in components:
    # Pick the IMC whose mean frequency is closest to this component's.
    ref_freq = float(np.mean(comp['true_IF']))
    idx = int(np.argmin(np.abs(imc_freqs - ref_freq)))
    comp['imc_index'] = idx
    comp['imc'] = IMC[idx]
    comp['result'] = analyze(IMC[idx])

    r = comp['result']
    true_IF_slice = comp['true_IF'][r['sl']]
    mae = np.mean(np.abs(true_IF_slice - r['est_IF']))
    print(f"{comp['name']:>14s}: IMC #{idx} (mean {imc_freqs[idx]:.2f} Hz), "
          f"mean |true IF - est IF| = {mae:.4f} Hz")

# --- Plot ------------------------------------------------------------------
# Metrics above use the full analysis window; the plots zoom into the first few
# seconds so the individual oscillations stay legible.
PLOT_WINDOW = 3.0  # seconds

fig, axes = plt.subplots(2, len(components), figsize=(6 * len(components), 7),
                          squeeze=False)

for col, comp in enumerate(components):
    r = comp['result']
    m = r['t'] - r['t'][0] <= PLOT_WINDOW

    ax = axes[0][col]
    ax.plot(r['t'][m], r['orig'][m], 'b', label=f"IMC #{comp['imc_index']}")
    ax.plot(r['t'][m], r['reconstr'][m], 'r--', label='JADE reconstruction')
    ax.plot(r['t'][m], comp['true_sig'][r['sl']][m], 'k', alpha=0.35,
            label='true component')
    ax.set_title(f"{comp['name']} - reconstruction")
    ax.set_ylabel('amplitude')
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[1][col]
    ax.plot(r['t'][m], comp['true_IF'][r['sl']][m], 'k', label='true IF')
    ax.plot(r['t'][m], r['est_IF'][m], 'r--', label='JADE IF estimate')
    ax.set_title(f"{comp['name']} - instantaneous frequency")
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')
    ax.legend(fontsize=8)
    ax.grid(True)

fig.tight_layout()
plt.savefig('jade_example.png', dpi=300)
print("saved jade_example.png")
