"""DPSS taper management for the multitaper PSD estimate.

DPSS taper computation is the O(N^2)+ eigendecomposition bottleneck in
multitaper spectral estimation. Only a handful of distinct (window_length, nw,
kspec) combos ever occur across the whole sweep, so each Pool worker is seeded
with every taper set up front (computed once in the main process, handed over
via the Pool initializer) instead of every worker recomputing its own copy
lazily on first use.
"""
from multitaper.utils import dpss


def _dpss_nw_kspec(window_length_sec, target_resolution_hz=0.1):
    nw = max(window_length_sec * target_resolution_hz / 3, 2)
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
