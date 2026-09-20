import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from iaf_compare.plot_style import FIG_WIDTH

# pgf.texsystem defaults to xelatex, which isn't installed -- pdflatex is.
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

fs        = 256
T         = 50
t         = np.arange(0, T, 1/fs)
N         = len(t)
np.random.seed(42)

f_carrier = 10.5;  delta_f = 0.5;  f_mod = 0.05
noise_std = 0.30

true_freq = f_carrier + delta_f * np.sin(2 * np.pi * f_mod * t)
phase     = 2 * np.pi * (f_carrier * t
            - (delta_f / (2*np.pi*f_mod)) * np.cos(2*np.pi*f_mod*t))
signal = np.cos(phase) + noise_std * np.random.randn(N)

def ar_dominant_freq(ar_coeffs, fs, f_min=8.0, f_max=14.0):
    poly  = np.concatenate([[1], -ar_coeffs])
    roots = np.roots(poly)
    freqs = np.angle(roots) * fs / (2*np.pi)
    mask  = (np.imag(roots) > 0) & (freqs >= f_min) & (freqs <= f_max)
    upper = roots[mask]
    if len(upper) == 0:
        return np.nan
    return np.angle(upper[np.argmax(np.abs(upper))]) * fs / (2*np.pi)

def run_arma_kalman(signal, p, q, sigma_w2, sigma_e2, fs):
    """
    ARMA(p,q) Kalman filter.

    Why q>0 matters:
      z_t = s_t + n_t, where s_t is AR(p) and n_t is white noise.
      Applying the AR operator A(z) to z_t gives:
          A(z) z_t = eps_t + A(z) n_t
      A(z) n_t is MA(p) colored noise.  The MA terms b_k let the
      model explicitly represent this coloring, so the AR coefficients
      (and hence the poles) are no longer forced to absorb the noise —
      they can sit at the true signal frequency unbiased.

    State  : theta = [a_1,...,a_p, b_1,...,b_q]
    Obs eq : z_t = phi_t^T theta + e_t
    phi_t  = [z_{t-1},...,z_{t-p}, e_{t-1},...,e_{t-q}]
    State eq: theta_{t+1} = theta_t + w_t  (random walk)
    """
    state_dim = p + q
    theta     = np.zeros(state_dim)
    P         = np.eye(state_dim) * 1e3
    past_e    = np.zeros(max(q, 1))   # ring buffer for past prediction errors
    freq      = np.full(len(signal), np.nan)

    for i in range(p, len(signal)):
        phi_ar = signal[[i-1-j for j in range(p)]]
        phi_ma = past_e[:q] if q > 0 else np.array([])
        phi    = np.concatenate([phi_ar, phi_ma])

        P_pred = P + sigma_w2 * np.eye(state_dim)
        S      = phi @ P_pred @ phi + sigma_e2
        k      = P_pred @ phi / S
        e      = signal[i] - phi @ theta
        theta  = theta + k * e
        P      = (np.eye(state_dim) - np.outer(k, phi)) @ P_pred

        if q > 0:
            past_e = np.roll(past_e, 1)
            past_e[0] = e

        freq[i] = ar_dominant_freq(theta[:p], fs)
    return freq

sigma_w2 = 1e-6
sigma_e2 = noise_std**2
skip     = int(fs * 1.5)

configs = [
    (6, 0, 'AR(6)       q=0  no MA terms'),
    (2, 2, 'ARMA(2,2)   theoretically optimal for single sinusoid + white noise'),
    (6, 2, 'ARMA(6,2)   paper choice, suitable for multi-component EEG'),
    (6, 6, 'ARMA(6,6)   p=q fully models white noise structure'),
]
colors = ['#e74c3c', '#27ae60', '#2980b9', '#8e44ad']

results = []
for p, q, label in configs:
    freq = run_arma_kalman(signal, p, q, sigma_w2, sigma_e2, fs)
    bias = np.nanmean(freq[skip:] - true_freq[skip:])
    std  = np.nanstd(freq[skip:]  - true_freq[skip:])
    results.append((freq, bias, std, label))
    print(f"ARMA({p},{q})  bias={bias:+.3f} Hz  std={std:.3f} Hz")

# ---- Plot ----
fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_WIDTH * 9 / 16))

axes[0].plot(t, signal, color='steelblue', lw=0.5, alpha=0.8)
axes[0].set_title(f'FM signal  —  noise_std={noise_std}  (SNR ≈ 10 dB)', fontsize=11)
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

axes[1].plot(t, true_freq, 'k-', lw=2.5, label='True frequency', zorder=5)
for (freq, bias, std, label), color in zip(results, colors):
    name = label.split('  ')[0]
    axes[1].plot(t, freq, '-', lw=1.1, alpha=0.8, color=color,
                 label=f'{name}  →  bias={bias:+.2f} Hz  std={std:.2f} Hz')

axes[1].set_ylim([f_carrier - delta_f*4, f_carrier + delta_f*4])
axes[1].set_title(
    'Adding MA terms: why they reduce bias\n'
    r'$z_t=s_t+n_t$ is ARMA not AR  $\Rightarrow$  '
    r'MA part absorbs $A(z)\,n_t$, freeing AR poles to sit at true frequency',
    fontsize=11)
axes[1].set_ylabel('Frequency [Hz]'); axes[1].set_xlabel('Time [s]')
axes[1].legend(fontsize=10)
axes[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.9)

plt.tight_layout()
plt.savefig('arma_kalman_fm.pdf', bbox_inches="tight")
plt.savefig('arma_kalman_fm.pgf', bbox_inches="tight")