import numpy as np
import matplotlib.pyplot as plt
from fooof import FOOOF
from scipy import stats
from scipy.signal import argrelmin, savgol_filter
from scipy.ndimage import center_of_mass

np.random.seed(0)

F_MIN = 7
F_MAX = 14
FREQ_RANGE = [1.0, 30.0]
SG_WINDOW = 11
SG_POLY = 3
PINK_MAX_R2 = 0.9

def main():
    fs = 10000.0
    signal_length = int(5 * fs)
    window_length = int(3 * fs)
    f0 = 10.0

    t = np.arange(signal_length) / fs
    signal = 0.25 * np.sin(2 * np.pi * f0 * t) + 0.23 * np.sin(2 * np.pi * (f0+0.5) * t) +pink_noise(signal_length, fs)

    results = {
        'stupid_max': [],
        'parabolic_max': [],
        'foof': [],
        'philistine': [],
        'combine': []
    }

    for start in range(1, signal_length-window_length, 100):
        window = signal[start:start + window_length]
        freq = np.fft.rfft(window, window_length)
        freq_bins = np.fft.rfftfreq(window_length, 1/fs)

        results['stupid_max'].append(stupid_max(freq, freq_bins))
        results['parabolic_max'].append(parabolic_max(freq, freq_bins))
        results['foof'].append(foof(freq, freq_bins))
        results['philistine'].append(philistine_iaf(freq, freq_bins))
        results['combine'].append(combine_algo(freq, freq_bins))

    freq_full = np.fft.rfft(signal)
    freq_bins_full = np.fft.rfftfreq(signal_length, 1/fs)

    plot_results(t[0:signal_length-window_length], signal, freq_full, freq_bins_full, results, f0, fs)


def plot_results(time_vec, signal, freq, freq_bins, results, true_freq, fs):
    adapt_time = time_vec[::100]
    resolution = freq_bins[1] - freq_bins[0]  # Hz per bin, not number of bins
    freq_mask = freq_bins <= 20

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flat

    # Signal
    ax1.plot(np.arange(len(signal)) / fs, signal)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Signal')

    # Spectrum
    ax2.plot(freq_bins[freq_mask], np.abs(freq[freq_mask]))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('FFT Magnitude')

    # Error over time
    for label, vals in results.items():
        errors = np.abs(np.array(vals) - true_freq)
        ax3.plot(adapt_time, errors, label=label)
    ax3.axhline(0, color='k', linestyle='--')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Absolute Error (Hz)')
    ax3.set_title('Error Over Time')
    ax3.legend()

    # Error distribution
    for label, vals in results.items():
        errors = np.abs(np.array(vals) - true_freq)
        ax4.hist(errors, alpha=0.6, label=label, bins=50)
    ax4.set_xlabel('Absolute Error (Hz)')
    ax4.set_ylabel('Count')
    ax4.set_title('Error Distribution per Algorithm')
    ax4.legend()

    plt.tight_layout()
    plt.show()

def pink_noise(N, fs=1.0):
    # Step 1: white noise
    x = np.random.randn(N)

    # Step 2: FFT
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # Step 3: avoid division by zero at f=0
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0

    # scale ~ 1/sqrt(f)
    scaling = 1 / np.sqrt(freqs)
    X = X * scaling

    # Step 4: inverse FFT
    y = np.fft.irfft(X, n=N)

    # normalize (optional but practical)
    y = y / np.std(y)

    return y

def stupid_max(freq, freq_bins):
    band = (freq_bins >= F_MIN) & (freq_bins <= F_MAX)
    mag = np.abs(freq[band])

    max_bin = int(np.argmax(mag))

    freq_bins_band = freq_bins[band]

    return freq_bins_band[max_bin]

def foof(freq, freq_bins):
    fm = FOOOF(peak_width_limits=[0.1, 4.0], min_peak_height=0.0,
               peak_threshold=2., max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    fm.fit(freq_bins, np.abs(freq)**2, FREQ_RANGE)

    if fm.n_peaks_ == 0:
        return 0

    alpha_peaks = [p for p in fm.peak_params_ if F_MIN <= p[0] <= F_MAX]
    return max(alpha_peaks, key=lambda p: p[1])[0] if alpha_peaks else 0

def parabolic_max(freq, freq_bins):
    band = (freq_bins >= F_MIN) & (freq_bins <= F_MAX)
    mag = np.abs(freq[band])
    max_bin = int(np.argmax(mag))
    freq_bins_band = freq_bins[band]

    if 0 < max_bin < (mag.size - 1):
        y1, y2, y3 = mag[max_bin - 1], mag[max_bin], mag[max_bin + 1]
        denom = (y1 - 2*y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            bin_hz = freq_bins_band[1] - freq_bins_band[0]
            return freq_bins_band[max_bin] + delta * bin_hz

    return freq_bins_band[max_bin]

def _auto_detect_edges(freqs, psd_flat, fmin_hint, fmax_hint):
    fmin_b = fmin_hint if fmin_hint is not None else 5.0
    fmax_b = fmax_hint if fmax_hint is not None else 15.0

    search = (freqs >= fmin_b) & (freqs <= fmax_b)
    f_s = freqs[search]
    p_s = psd_flat[search]

    wl = min(SG_WINDOW, len(p_s))
    if wl > SG_POLY and len(p_s) >= wl:
        p_s = savgol_filter(p_s, window_length=wl, polyorder=SG_POLY)

    fmin_out = fmin_hint
    fmax_out = fmax_hint

    if fmin_out is None:
        mins = argrelmin(p_s[f_s < 10])[0]
        fmin_out = f_s[f_s < 10][mins[-1]] if mins.size > 0 else F_MIN

    if fmax_out is None:
        mins = argrelmin(p_s[f_s > 10])[0]
        fmax_out = f_s[f_s > 10][mins[0]] if mins.size > 0 else F_MAX

    return fmin_out, fmax_out

def combine_algo(freq, freq_bins):
    band = (freq_bins >= FREQ_RANGE[0]) & (freq_bins <= FREQ_RANGE[1])
    psd = np.abs(freq[band])**2
    freqs = freq_bins[band]

    resolution = freqs[1] - freqs[0]
    sav_gol_window_length = int(2 / resolution)

    fm = FOOOF(peak_width_limits=(1.0, 8.0), max_n_peaks=6, min_peak_height=0.0,
               peak_threshold=2.0, aperiodic_mode="fixed", verbose=False)
    fm.fit(freqs, psd, FREQ_RANGE)

    offset, exponent = fm.aperiodic_params_
    aperiodic = offset - exponent * np.log10(freqs)
    residual = np.log10(psd) - aperiodic
    psd_flat = np.power(10, residual)

    psd_smooth = savgol_filter(psd_flat, window_length=sav_gol_window_length, polyorder=3)
    alpha_mask = (freqs >= F_MIN) & (freqs <= F_MAX)

    return freqs[alpha_mask][np.argmax(psd_smooth[alpha_mask])]

def philistine_iaf(freq, freq_bins):
    band = (freq_bins >= FREQ_RANGE[0]) & (freq_bins <= FREQ_RANGE[1])
    psd = np.abs(freq[band])**2
    freqs = freq_bins[band]
    resolution = freqs[1] - freqs[0]

    sav_gol_window_length = int(2 / resolution)
    fmin, fmax = F_MIN, F_MAX

    if fmin is None or fmax is None:
        fmin_bound = fmin if fmin is not None else 5
        fmax_bound = fmax if fmax is not None else 15

        alpha_search = (freqs >= fmin_bound) & (freqs <= fmax_bound)
        freqs_search = freqs[alpha_search]
        psd_search = savgol_filter(psd[alpha_search], window_length=psd[alpha_search].shape[0], polyorder=10)

        if fmin is None:
            mins = argrelmin(psd_search[freqs_search < 10])[0]
            fmin = freqs_search[freqs_search < 10][mins[-1]] if mins.size > 0 else F_MIN

        if fmax is None:
            mins = argrelmin(psd_search[freqs_search > 10])[0]
            fmax = freqs_search[freqs_search > 10][mins[0]] if mins.size > 0 else F_MAX

    psd_smooth = savgol_filter(psd, window_length=sav_gol_window_length, polyorder=3)
    alpha_band = (freqs >= fmin) & (freqs <= fmax)

    eps = 1e-12
    slope, intercept, r, p, se = stats.linregress(np.log(freqs), np.log(np.maximum(psd_smooth, eps)))

    if r**2 > PINK_MAX_R2:
        return None

    paf_idx = np.argmax(psd_smooth[alpha_band])
    paf = freqs[alpha_band][paf_idx]

    cog_idx = center_of_mass(psd_smooth[alpha_band])
    try:
        cog_idx = int(np.round(cog_idx[0]))
        cog = freqs[alpha_band][cog_idx]
    except (ValueError, IndexError):
        cog = None

    return paf


if __name__ == "__main__":
    main()