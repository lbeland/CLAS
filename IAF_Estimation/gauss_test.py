"""
Compare two ways of adding an oscillatory (Gaussian) peak on top of an
aperiodic 1/f (pink noise) background:

  A) Gaussian defined in LINEAR power space, then everything log-transformed
         P(f) = L(f) * O(f),   O(f) = 1 + a * exp(-(f-c)^2 / (2*w^2))

  B) Gaussian defined in LOG power space (FOOOF-style), then transformed back
         log10(P(f)) = log10(L(f)) + G(f),   G(f) = a * exp(-(f-c)^2 / (2*w^2))
         P(f) = L(f) * 10**G(f)

Same nominal amplitude / center / width parameters are used for both, so the
plots make the structural difference visible: the log-space Gaussian becomes
narrower / more peaked and asymmetric once transformed back to linear power,
while the linear-space Gaussian stays a "true" Gaussian bump in linear space.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Aperiodic (pink noise) background: L(f) = f^(-chi) * 10^b
# ----------------------------
chi = 1.2
b = 0.3  # offset

f = np.linspace(1, 40, 2000)  # Hz, avoid f=0
L = f ** (-chi) * 10 ** b

# ----------------------------
# Shared Gaussian parameters
# ----------------------------
c = 10.0     # center frequency (Hz) - alpha peak
w = 1.5      # width (std), same shape parameter used for both

gauss_shape = np.exp(-((f - c) ** 2) / (2 * w ** 2))

# A) Linear-space Gaussian:  P = L * (1 + a_linear * gauss_shape)
a_linear = 4.0
O_linear = 1 + a_linear * gauss_shape
P_linear_model = L * O_linear

# B) Log-space Gaussian:  log10(P) = log10(L) + a_log * gauss_shape
# Amplitude a_log is chosen so that the resulting peak reaches the SAME
# maximum linear power factor as the linear-space model above, i.e.
#   10^a_log = 1 + a_linear  =>  a_log = log10(1 + a_linear)
# This makes the two peaks comparable in linear space (same max height),
# so the plot isolates the effect of the *shape*, not an arbitrary
# amplitude mismatch.
a_log = np.log10(1 + a_linear)
G_log = a_log * gauss_shape
P_log_model = L * (10 ** G_log)

# ----------------------------
# Direct overlay: both models on top of each other, same axes
# ----------------------------
fig2, axes2 = plt.subplots(2, 3, figsize=(11, 4.5))

ax = axes2[0,0]
ax.plot(f, L, color="gray", lw=1.5, ls="--", label=r"$L(f)$ (aperiodic)")
ax.plot(f, P_linear_model, color="tab:blue", lw=2, label="linear-space Gaussian")
ax.plot(f, P_log_model, color="tab:red", lw=2, ls="-.", label="log-space Gaussian")
ax.axvline(c, color="k", lw=0.5, alpha=0.3)
ax.set_title("Linear power space")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (linear)")
ax.legend(fontsize=8)

ax = axes2[1,0]
ax.plot(f, P_linear_model/L, color="tab:blue", lw=2, ls="-.", label="linear-space Gaussian")
ax.plot(f, P_log_model/L, color="tab:red", lw=2, ls="-.", label="log-space Gaussian")
ax.axvline(c, color="k", lw=0.5, alpha=0.3)
ax.set_title("Linear power space")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (linear)")
ax.legend(fontsize=8)

ax = axes2[0,1]
ax.plot(f, np.log10(L), color="gray", lw=1.5, ls="--", label=r"$\log_{10}L(f)$")
ax.plot(f, np.log10(P_linear_model), color="tab:blue", lw=2, label="linear-space Gaussian")
ax.plot(f, np.log10(P_log_model), color="tab:red", lw=2, ls="-.", label="log-space Gaussian")
ax.axvline(c, color="k", lw=0.5, alpha=0.3)
ax.set_title(r"$\log_{10}$ power space")
ax.set_xlabel("Frequency (Hz)")
ax.legend(fontsize=8)

ax = axes2[1,1]
ax.plot(f, np.log10(P_linear_model)-np.log10(L), color="tab:blue", lw=2, ls="-.", label="linear-space Gaussian")
ax.plot(f, np.log10(P_log_model)-np.log10(L), color="tab:red", lw=2, ls="-.", label="log-space Gaussian")
ax.axvline(c, color="k", lw=0.5, alpha=0.3)
ax.set_title(r"$\log_{10}$ power space")
ax.set_xlabel("Frequency (Hz)")
ax.legend(fontsize=8)

ax = axes2[0,2]
ax.plot(np.log10(f), np.log10(L), color="gray", lw=1.5, ls="--", label=r"$\log_{10}L(f)$")
ax.plot(np.log10(f), np.log10(P_linear_model), color="tab:blue", lw=2, label="linear-space Gaussian")
ax.plot(np.log10(f), np.log10(P_log_model), color="tab:red", lw=2, ls="-.", label="log-space Gaussian")
ax.axvline(np.log10(c), color="k", lw=0.5, alpha=0.3)
ax.set_title(r"$\log_{10}-\log_{10}$ power space")
ax.set_xlabel("Frequency (Hz)")
ax.legend(fontsize=8)

ax = axes2[1,2]
ax.plot(np.log10(f), np.log10(P_linear_model)-np.log10(L), color="tab:blue", lw=2, ls="-.", label="linear-space Gaussian")
ax.plot(np.log10(f), np.log10(P_log_model)-np.log10(L), color="tab:red", lw=2, ls="-.", label="log-space Gaussian")
ax.axvline(np.log10(c), color="k", lw=0.5, alpha=0.3)
ax.set_title(r"$\log_{10}-\log_{10}$ power space")
ax.set_xlabel("Frequency (Hz)")
ax.legend(fontsize=8)

fig2.suptitle("Same peak height in linear space, but different shape/width", fontsize=11)
fig2.tight_layout(rect=[0, 0, 1, 0.92])


# ----------------------------
# Numeric comparison: peak height and FWHM in linear space
# ----------------------------
peak_idx = np.argmin(np.abs(f - c))

def fwhm_linear(power_over_baseline_minus1, f):
    """FWHM of the excess-over-baseline factor (O(f)-1) in linear space."""
    half_max = power_over_baseline_minus1.max() / 2
    above = power_over_baseline_minus1 >= half_max
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]
    return f[idx[-1]] - f[idx[0]]

excess_linear = O_linear - 1
excess_log = (P_log_model / L) - 1

print("\nPeak factor O(f) = P(f)/L(f) at center (should match by construction):")
print(f"  linear-space Gaussian model: {O_linear[peak_idx]:.3f}")
print(f"  log-space Gaussian model:    {(P_log_model / L)[peak_idx]:.3f}")

print("\nFWHM of the excess factor (O(f)-1) in linear space:")
print(f"  linear-space Gaussian model: {fwhm_linear(excess_linear, f):.3f} Hz")
print(f"  log-space Gaussian model:    {fwhm_linear(excess_log, f):.3f} Hz")

plt.show()