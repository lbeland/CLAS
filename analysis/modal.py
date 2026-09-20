"""
MODAL - Multiple Oscillation Detection Algorithm (Python port)
================================================================
Ported from MODAL.m (Andrew J Watrous, October 2017).

Provides the instantaneous frequency of a (neural) signal in adaptively
identified oscillatory bands, i.e. bands in which power exceeds a global
(and optionally local) 1/f fit of the signal "background".

Only `frequency_sliding` is returned, per request -- `bands`, `bandpow`,
and `bandphases` are computed internally (the algorithm depends on them)
but are not returned.

Reference for the "frequency sliding" method:
    Cohen MX (2014) Fluctuations in oscillation frequency control
    spike timing and coordinate neural networks. J Neurosci.
    http://mikexcohen.com/data/Cohen2014_freqslide.pdf

Reference for adaptive band identification:
    Lega BC, Jacobs J, Kahana M (2012) Human hippocampal theta
    oscillations and the formation of episodic memories. Hippocampus.

Dependencies: numpy, scipy, statsmodels
"""

import warnings

import numpy as np
from scipy.signal import firls, filtfilt, fftconvolve, hilbert, medfilt
from scipy.ndimage import label
import statsmodels.api as sm


def _morlet_wavelet_power(signal, wavefreqs, srate, wavecycles=6):
    """Substitute for the Kahana-lab "multiphasevec2" function: a
    (frequencies x time) power matrix via a bank of complex Morlet wavelets.
    """
    n = len(signal)
    pow_out = np.zeros((len(wavefreqs), n))

    for i, f in enumerate(wavefreqs):
        sigma_t = wavecycles / (2 * np.pi * f)
        half_len = int(np.ceil(3.5 * sigma_t * srate))
        if 2 * half_len + 1 > n:
            # Frequency too low to resolve within this signal length
            warnings.warn(
                f"_morlet_wavelet_power: frequency {f:.4g} Hz needs a wavelet "
                f"longer than the {n}-sample signal; setting power to NaN."
            )
            pow_out[i, :] = np.nan
            continue
        t = np.arange(-half_len, half_len + 1) / srate
        wavelet = np.exp(2j * np.pi * f * t) * np.exp(-t ** 2 / (2 * sigma_t ** 2))
        # normalize wavelet energy
        wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet) ** 2))
        # FFT-based, unlike the direct-form np.convolve this was ported from:
        # a low-frequency wavelet's kernel can run to tens of thousands of taps
        conv = fftconvolve(signal, wavelet, mode="same")
        pow_out[i, :] = np.abs(conv) ** 2

    return pow_out


def _robustfit(x, y):
    """Robust linear fit matching MATLAB's robustfit(x, y) default (IRLS with
    a bisquare/Tukey weight function). Returns (intercept, slope) for y ~ b0 + b1*x.
    """
    good = ~np.isnan(x) & ~np.isnan(y)
    X = sm.add_constant(x[good])
    model = sm.RLM(y[good], X, M=sm.robust.norms.TukeyBiweight())
    result = model.fit()
    b0, b1 = result.params[0], result.params[1]
    return b0, b1


def _dsearchn(x, xi):
    """Nearest-neighbor lookup, equivalent to MATLAB's dsearchn(x, xi) for a
    1-D sorted vector x and a scalar/array xi."""
    x = np.asarray(x)
    idx = np.searchsorted(x, xi)
    idx = np.clip(idx, 1, len(x) - 1)
    left = x[idx - 1]
    right = x[idx]
    idx = idx - (np.abs(xi - left) < np.abs(xi - right))
    return idx


def _medfilt1(x, n):
    """1-D median filter approximating MATLAB's medfilt1(x, n); even orders
    are rounded up to the next odd value, since scipy.signal.medfilt requires
    odd kernels."""
    n = int(round(n))
    if n < 1:
        n = 1
    if n % 2 == 0:
        n += 1
    if n > len(x):
        n = len(x) if len(x) % 2 == 1 else len(x) - 1
        if n < 1:
            n = 1
    return medfilt(x, kernel_size=n)


def _get_bands(wavefreqs, pow_mat):
    """Key Step #1: adaptively identify oscillatory bands from a 1/f fit.

    Parameters
    ----------
    wavefreqs : 1D array of sampled frequencies (not log-transformed)
    pow_mat : (frequencies x time) power matrix (not log-transformed)

    Returns
    -------
    freq_bands : (n_bands x 2) array of [lower, upper] band edges (Hz)
    bandidx : list of arrays, indices into wavefreqs for each band
    bandpow : (n_bands x time) mean log-power per band
    """
    fz = np.log(wavefreqs)
    mean_pow = np.log(np.nanmean(pow_mat, axis=1))

    b0, b1 = _robustfit(fz, mean_pow)
    fit_line = b0 + b1 * fz

    above_1f = (mean_pow - fit_line) > 0
    labeled, n_labels = label(above_1f)

    freq_bands = []
    bandidx = []
    bandpow_list = []

    for i_band in range(1, n_labels + 1):
        idx = np.where(labeled == i_band)[0]
        if len(idx) > 1:  # must be an actual band, not a single point-frequency
            freq_bands.append([wavefreqs[idx.min()], wavefreqs[idx.max()]])
            bandidx.append(idx)
            bandpow_list.append(np.log(np.mean(pow_mat[idx, :], axis=0)))

    if len(freq_bands) == 0:
        return np.zeros((0, 2)), [], np.zeros((0, pow_mat.shape[1]))

    return np.array(freq_bands), bandidx, np.array(bandpow_list)


def _fit_one_over_f_windows(frequency_sliding, wavefreqs, pow_mat, bandidx):
    """Key Step #3: NaN out frequency-sliding estimates below a *local* 1/f
    fit (computed on a smaller time window than the global fit).

    Parameters
    ----------
    frequency_sliding : (n_bands x window_time) FS estimates for this window
    wavefreqs : 1D array of sampled frequencies
    pow_mat : (frequencies x window_time) power for this window
    bandidx : list of arrays, indices into wavefreqs for each band

    Returns
    -------
    frequency_sliding with sub-threshold values replaced by NaN
    """
    fz = np.log(wavefreqs)
    local_mean_pow = np.log(np.nanmean(pow_mat, axis=1))
    b0, b1 = _robustfit(fz, local_mean_pow)
    local_fit_line = b0 + b1 * fz

    logpow = np.log(pow_mat)  # frequencies x time
    fitpow = np.tile(local_fit_line[:, None], (1, logpow.shape[1]))
    powdiff = logpow - fitpow
    threshpow = powdiff > 0  # frequencies x time boolean

    tmp_fs = frequency_sliding.copy()

    for i_b in range(len(bandidx)):
        idx1 = np.where(~np.isnan(frequency_sliding[i_b, :]))[0]
        if len(idx1) > 0:
            fswf = _dsearchn(wavefreqs, frequency_sliding[i_b, idx1])
            threshvalz = threshpow[fswf, idx1]
            tmp_fs[i_b, idx1[threshvalz == 0]] = np.nan
        else:
            tmp_fs[i_b, :] = np.nan

    return tmp_fs


def modal(signal, params):
    """Multiple Oscillation Detection Algorithm (MOD-AL).

    Parameters
    ----------
    signal : 1D array-like
        Signal to analyze (any neural timeseries).
    params : dict
        'srate' : sampling rate (Hz).
        'wavefreqs' : 1D array of frequencies for background fitting
            (recommend max frequency >= 30 Hz for a good 1/f fit).
        'bad_data' (optional) : boolean array, same length as signal;
            True == excluded from calculations.
        'local_winsize_sec' (optional) : window sizes (seconds) for local 1/f
            fitting, e.g. [1, 5, 10]. Defaults to a single 10-second window;
            pass [] to skip local thresholding (return FS for all timepoints).
        'crop_fs' (optional) : bool, crop estimates outside the detected band
            (default True).
        'wavecycles' (optional) : wavelet cycles (default 6).

    Returns
    -------
    frequency_sliding : (n_bands x n_samples) ndarray (float32)
        Instantaneous frequency of the signal in each detected band.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim > 1:
        signal = signal.reshape(-1)

    srate = params["srate"]
    wavefreqs = np.asarray(params["wavefreqs"], dtype=float)

    if "local_winsize_sec" in params and params["local_winsize_sec"] is not None:
        wins = np.asarray(params["local_winsize_sec"], dtype=float) * srate
    else:
        wins = np.array([srate * 10.0])  # default: single 10-second window

    wavecycles = params.get("wavecycles", 6)
    crop_fs = params.get("crop_fs", True)

    # mean-center the signal so Hilbert transform / power estimates are valid
    signal = signal - np.nanmean(signal)
    n_samples = len(signal)

    # Extract (frequencies x time) power matrix
    pow_mat = _morlet_wavelet_power(signal, wavefreqs, srate, wavecycles)

    # Handle bad data: NaN out power during bad times
    if "bad_data" in params and params["bad_data"] is not None:
        bad_idx = np.where(np.asarray(params["bad_data"]) == 1)[0]
        pow_mat[:, bad_idx] = np.nan

    # Key Step #1: adaptive band identification via global 1/f fit
    bands, bandidx, _bandpow = _get_bands(wavefreqs, pow_mat)
    n_bands = bands.shape[0]

    if n_bands == 0:
        return np.array(np.nan)

    # Key Step #2: frequency sliding per band
    FS = np.full((n_bands, n_samples), np.nan)
    trans_width = 0.15
    ideal_response = [0, 0, 1, 1, 0, 0]
    time_wins = np.array([0.05, 0.2, 0.4])  # seconds, from MX Cohen
    orders = time_wins * srate
    numchunks = 10

    for i_band in range(n_bands):
        f_lo, f_hi = bands[i_band, 0], bands[i_band, 1]

        filt_freq_bounds = np.array(
            [0, (1 - trans_width) * f_lo, f_lo, f_hi, f_hi * (1 + trans_width), srate / 2]
        ) / (srate / 2)

        filt_order = int(round(2 * (srate / f_lo)))
        numtaps = filt_order + 1
        if numtaps % 2 == 0:  # firls requires an odd number of taps
            numtaps += 1

        filter_weights = firls(numtaps, filt_freq_bounds, ideal_response)
        filtered_signal = filtfilt(filter_weights, [1.0], signal)

        analytic = hilbert(filtered_signal)
        angle_hilbert = np.angle(analytic)
        # bandphases would be stored here (not returned)

        frompaper = srate * np.diff(np.unwrap(angle_hilbert)) / (2 * np.pi)
        frompaper = np.append(frompaper, np.nan)  # diff loses a sample

        chunks = np.floor(np.linspace(0, len(frompaper) - 1, numchunks)).astype(int)

        meds = np.zeros((len(orders), len(frompaper)))
        for i_win, order in enumerate(orders):
            for i_chunk in range(1, numchunks):
                chunkidx = np.arange(chunks[i_chunk - 1], chunks[i_chunk])
                if len(chunkidx) > 0:
                    meds[i_win, chunkidx] = _medfilt1(frompaper[chunkidx], order)

        median_of_meds = np.median(meds, axis=0)

        # Key Step #4: NaN out estimates outside the detected band (phase slips)
        if crop_fs:
            below_idx = median_of_meds < f_lo
            above_idx = median_of_meds > f_hi
            outside_idx = below_idx | above_idx
            median_of_meds[outside_idx] = np.nan

        FS[i_band, :] = median_of_meds

    # Optional Key Step #3: local 1/f thresholding; an empty wins skips it,
    # matching MODAL.m's `for iW = 1:length(wins)` never executing
    if len(wins) == 0:
        frequency_sliding = FS
    else:
        frequency_sliding_windows = np.tile(FS[:, :, None], (1, 1, len(wins)))

        for i_w, winsize in enumerate(wins):
            winsize = int(winsize)
            for i_win_start in range(0, n_samples, winsize):
                windex = np.arange(i_win_start, min(i_win_start + winsize + 1, n_samples))

                n_nan = np.sum(np.isnan(pow_mat[:, windex]))
                if n_nan < (len(windex) * len(wavefreqs)):  # skip if window is all-NaN
                    frequency_sliding_windows[:, windex, i_w] = _fit_one_over_f_windows(
                        FS[:, windex], wavefreqs, pow_mat[:, windex], bandidx
                    )
                else:
                    frequency_sliding_windows[:, windex, i_w] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            frequency_sliding = np.nanmean(frequency_sliding_windows, axis=2)

    frequency_sliding = frequency_sliding.astype(np.float32)

    return frequency_sliding


if __name__ == "__main__":
    # small smoke test with synthetic data: a noisy signal with a strong
    # ~10 Hz oscillation embedded, to confirm the pipeline runs end-to-end.
    np.random.seed(0)
    srate = 500
    t = np.arange(0, 20, 1 / srate)
    sig = np.sin(2 * np.pi * 12.4 * t) * 3 + np.random.randn(len(t))

    params = {
        "srate": srate,
        "wavefreqs": np.arange(0.1, 30, 1.0),
        "local_winsize_sec": [10],
        "wavecycles": 6,
        "crop_fs": True,
    }

    fs = modal(sig, params)
    print("frequency_sliding shape:", fs.shape)
    print(np.nanmean(fs, axis=1))  # should be close to 10 Hz for the strong oscillation