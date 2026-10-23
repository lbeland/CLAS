"""DPSS taper management for the multitaper PSD estimate.

DPSS taper computation is the O(N^2)+ eigendecomposition bottleneck in
multitaper spectral estimation. Only a handful of distinct (window_length, nw,
kspec) combos ever occur across the whole sweep, so each Pool worker is seeded
with every taper set up front (computed once in the main process, handed over
via the Pool initializer) instead of every worker recomputing its own copy
lazily on first use.
"""
import numpy as np
from scipy.signal import welch
from multitaper import MTSpec
from multitaper.utils import dpss


def _dpss_nw_kspec(window_length_sec, target_resolution_hz=0.4):
    nw = max(window_length_sec * target_resolution_hz, 2)
    kspec = int(2 * nw - 1)
    return nw, kspec


_DPSS_CACHE = {}


def _init_dpss_cache(precomputed):
    _DPSS_CACHE.update(precomputed)


def _cached_dpss(window_length, nw, kspec):
    key = (window_length, nw, kspec)
    if key not in _DPSS_CACHE:
        _DPSS_CACHE[key] = dpss(window_length, nw, kspec)
    return _DPSS_CACHE[key]


def precompute_dpss(window_length_secs, fs):
    """Compute the DPSS taper set for every distinct window length up front,
    keyed by (npts, nw, kspec) -- the key shape :func:`_cached_dpss` expects.
    Hand the result to ``Pool(initializer=_init_dpss_cache, initargs=(...,))``
    so each expensive eigendecomposition happens exactly once total."""
    precomputed = {}
    for wl_sec in set(window_length_secs):
        npts = int(wl_sec * fs)
        nw, kspec = _dpss_nw_kspec(wl_sec)
        precomputed[(npts, nw, kspec)] = dpss(npts, nw, kspec)
    return precomputed

def compute_spectra(window, window_length, fs, config):
    """Periodogram + Welch + multitaper PSDs for one window. window_length is
    taken as a separate argument rather than len(window) since the caller's
    intended window_length (== config["window_length_sec"] * fs) can exceed
    len(window) itself when the signal is shorter than the nominal window."""
    nperseg = int(min(window_length, 2 * fs))
    freq_bins_welch, psd_welch = welch(window, fs=fs, nperseg=nperseg, noverlap=None)

    freq_bins = np.fft.rfftfreq(window_length, 1 / fs)
    X = np.fft.rfft(window, n=window_length)
    # Periodogram PSD: |X|^2 / (fs * N) — matches Welch units (power per Hz)
    psd = (np.abs(X) ** 2) / (fs * window_length)
    psd[1:-1] *= 2  # Correct for dropping negative freqs in one-sided spectrum (except DC and Nyquist)

    nw, kspec = _dpss_nw_kspec(config["window_length_sec"])
    vn, lamb = _cached_dpss(window_length, nw, kspec)  # pre-seeded by the Pool initializer

    mt = MTSpec(window, nw=nw, kspec=kspec, dt=1 / fs, vn=vn, lamb=lamb)
    freq_mt, psd_mt = mt.rspec()

    return psd, psd_welch, psd_mt, freq_bins, freq_bins_welch, freq_mt
