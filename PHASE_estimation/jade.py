"""Python port of JADE_v3.m: instantaneous phase/frequency estimation via DTW.

This is a line-by-line translation of JADE_v3.m, preserving its algorithmic
structure, index arithmetic and default-behavior side effects (print/warn
messages). MATLAB is 1-based; to keep the numeric outputs identical to the
original, `zero_crossings` (and all internal cycle-boundary bookkeeping) is
kept in the *same 1-based sample-index convention* as MATLAB. If you index
into a numpy array with a value coming out of `zero_crossings`, subtract 1.

Please cite:
J. C. Mouli, D. Anderson, A. Cicone. On the Instantaneous Phase and
Frequency Estimation of a Non-stationary Signal. The JADE Algorithm.
Submitted. ArXiv https://arxiv.org/pdf/2604.14185

NOTE on two ported helpers that could not be verified bit-for-bit:
  * `_smooth_data` (MATLAB `smoothdata(x)`, used only when smooth=1):
    MATLAB's default automatic window-length selection for the 'movmean'
    method is not published in closed form, so this uses a simple adaptive
    moving-average heuristic instead. Only affects the smooth=1 option.
  * `_linear_spline_near_boundary` (used only when normalize=1): the MATLAB
    source for this helper was not present alongside JADE_v3.m in this
    repository, so it is reconstructed here (linear extrapolation of the
    correction curve out to the signal boundaries from the nearest detected
    peaks). Only affects the normalize=1 option.
Both default to off (smooth=0, normalize=0), so the default code path is a
faithful translation.
"""

import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks


def jade_v3(x, tt, Ts, bpi=1, N=0, smooth=0, ZC=None, templates=None,
            normalize=0, IF_N=0):
    """Estimate instantaneous phase and frequency using DTW.

    Required inputs:
        x, tt   Signal and its time vector (same length, tt strictly increasing).
        Ts      Sampling period.

    Optional inputs (see JADE_v3.m for full documentation):
        bpi, N, smooth, ZC, templates, normalize, IF_N

    Returns:
        IFvals, phasevals, zero_crossings, amplitudegram

    `zero_crossings` (and the internal breakpoint bookkeeping) use the same
    1-based sample-index convention as the original MATLAB code.
    """

    x = np.asarray(x, dtype=float).ravel()
    tt = np.asarray(tt, dtype=float).ravel()

    if not np.all(np.isfinite(x)):
        raise ValueError('JADE_v3:InputNotFinite x must contain only finite values.')
    if not np.all(np.isfinite(tt)):
        raise ValueError('JADE_v3:InputNotFinite tt must contain only finite values.')
    if not (np.isfinite(Ts) and Ts > 0):
        raise ValueError('JADE_v3:InvalidTs Ts must be a finite positive scalar.')

    bpi = _validate_nonneg_int(bpi, 'bpi')
    N = _validate_nonneg_int(N, 'N')
    IF_N = _validate_nonneg_int(IF_N, 'IF_N')
    smooth = _validate_binary(smooth, 'smooth')
    normalize = _validate_binary(normalize, 'normalize')

    if templates is None:
        templates = []
    elif not isinstance(templates, (list, tuple)):
        raise ValueError('JADE_v3:InvalidTemplates templates must be a list/cell array.')
    templates = list(templates)

    if x.size != tt.size:
        raise ValueError('JADE_v3:InputSizeMismatch x and tt must have the same '
                          'number of elements.')

    if x.size < 3:
        raise ValueError('JADE_v3:InputTooShort x and tt must contain at least '
                          'three samples.')

    if np.any(np.diff(tt) <= 0):
        raise ValueError('JADE_v3:InvalidTimeVector tt must be strictly increasing.')

    dtMedian = np.median(np.diff(tt))
    if abs(dtMedian - Ts) > max(1e-10, 1e-6 * Ts):
        warnings.warn('JADE_v3:SamplingPeriodMismatch Ts differs from the median '
                       'spacing of tt. Ts is used for phase/frequency calculations.')

    if bpi < 1:
        raise ValueError('JADE_v3:InvalidBreakpointSpacing bpi must be a positive '
                          'integer.')

    if ZC is None:
        ZC = np.array([])
    else:
        ZC = np.asarray(ZC, dtype=float).ravel()
        if not np.all(np.isfinite(ZC)):
            raise ValueError('JADE_v3:InvalidZC ZC must contain finite values.')

    if ZC.size == 0 and len(templates) > 0:
        raise ValueError('JADE_v3:TemplatesRequireZC A priori templates require '
                          'a priori zero-crossings (ZC).')

    # Optional amplitude normalization
    if normalize == 1:
        xs, _ = _scale_amplitudes(x)
        print('Removing AM using spline normalization scheme')
    else:
        xs = x

    # Optional smoothing for zero-crossing detection
    if smooth == 1:
        print('smoothing input data')
        xs_sm = _smooth_data(xs)
    else:
        xs_sm = xs

    # Zero-crossing detection
    if ZC.size == 0:
        prod = xs_sm[:-1] * xs_sm[1:]
        crossing_idx = np.where(prod <= 0)[0] + 1  # 1-based, matches MATLAB find()

        if crossing_idx.size:
            keep = np.ones(crossing_idx.size, dtype=bool)
            keep[1:] = np.diff(crossing_idx) > 1
            crossing_idx = crossing_idx[keep]

        zero_crossings = crossing_idx

        if zero_crossings.size == 0:
            raise ValueError('JADE_v3:NoZeroCrossings No zero crossings were detected.')
    else:
        print('using provided zero-crossings')
        zero_crossings = ZC

        if np.any(zero_crossings != _mround(zero_crossings)):
            raise ValueError('JADE_v3:InvalidZC ZC must contain integer sample indices.')

        if np.any(zero_crossings < 1) or np.any(zero_crossings > x.size):
            raise ValueError('JADE_v3:InvalidZC ZC contains indices outside the '
                              'input signal.')

        if np.any(np.diff(zero_crossings) <= 0):
            raise ValueError('JADE_v3:InvalidZC ZC must be strictly increasing.')

    zero_crossings = zero_crossings.astype(np.int64)

    if zero_crossings.size < 2:
        raise ValueError('JADE_v3:InsufficientZeroCrossings At least two zero '
                          'crossings are required.')

    numCycles = zero_crossings.size - 1

    if templates and len(templates) != numCycles:
        raise ValueError('JADE_v3:TemplateCountMismatch The number of templates '
                          f'({len(templates)}) must equal the number of '
                          f'zero-crossing intervals ({numCycles}).')

    # Main DTW loop
    phasegram = np.array([], dtype=float)
    amplitudegram = np.array([], dtype=float)

    for i in range(2, zero_crossings.size + 1):  # MATLAB-style 1-based i
        idx = i - 1  # 0-based position into zero_crossings

        int_ = int(zero_crossings[idx]) - 1
        prevint = int(zero_crossings[idx - 1])

        if int_ < prevint:
            raise ValueError('JADE_v3:InvalidCycle Zero-crossing indices do not '
                              'define a valid cycle.')

        cur_cycle = xs[prevint - 1:int_]
        distance = tt[int_ - 1] - tt[prevint - 1]

        if distance <= 0:
            raise ValueError('JADE_v3:InvalidCycleDuration Cycle duration must '
                              'be positive.')

        ft = 1.0 / (2.0 * distance)

        nTemplate = int(max(2, _mround(distance / Ts) + 1))
        templateTime = np.arange(nTemplate) * Ts

        if templateTime[-1] > distance:
            templateTime[-1] = distance

        if np.any(np.diff(templateTime) <= 0):
            templateTime = np.array([0.0, distance])

        # Template construction
        if not templates:
            signAtEnd = np.sign(xs_sm[int_ - 1])
            if signAtEnd == 0 and int_ > prevint:
                signAtEnd = np.sign(xs_sm[int_ - 2])

            if signAtEnd == 0:
                raise ValueError('JADE_v3:AmbiguousTemplateSign Could not '
                                  f'determine template sign for cycle {i - 1}.')

            unitTemplate = signAtEnd * np.sin(2 * np.pi * templateTime * ft)

            def temp_match(amp, unitTemplate=unitTemplate, cur_cycle=cur_cycle):
                return _dtw_distance(amp * unitTemplate, cur_cycle)

            if signAtEnd > 0:
                ampUpper = np.max(cur_cycle)
            else:
                ampUpper = abs(np.min(cur_cycle))

            ampUpper = max(ampUpper, np.finfo(float).eps)
            res = minimize_scalar(temp_match, bounds=(0, ampUpper), method='bounded')
            estAmp = res.x
            template = estAmp * unitTemplate

        else:
            template = np.asarray(templates[i - 2], dtype=float).ravel()

            if template.size == 0 or not np.all(np.isfinite(template)):
                raise ValueError('JADE_v3:InvalidTemplate Template '
                                  f'{i - 1} must be a nonempty finite numeric vector.')

            estAmp = np.max(np.abs(template))

            if estAmp == 0:
                warnings.warn(f'JADE_v3:ZeroTemplate Template {i - 1} has zero amplitude.')

        # Dynamic time warping
        _, ix, iy = _dtw(template, cur_cycle)

        ixn = iy.astype(float)
        iyn = ix.astype(float)

        if ixn.size == 0 or iyn.size == 0:
            raise ValueError('JADE_v3:EmptyDTWPath DTW returned an empty warping '
                              f'path for cycle {i - 1}.')

        iyn = iyn - iyn[0]

        # Phase estimation
        if N > 0:
            polynomial = _polyfit_zero(ixn, iyn, N)
            phi_est = np.zeros(N + 1)

            for j in range(1, N + 2):
                order = N + 1 - j
                phi_est[j - 1] = polynomial[j - 1] * (ft / Ts ** (order - 1))

            sampledphase = np.polyval(phi_est, templateTime)
            sampledphase = sampledphase - sampledphase[0]

            if phasegram.size:
                sampledphase = sampledphase + phasegram[-1]

        else:
            diffs = np.diff(ixn)
            changeIdx = np.concatenate([np.where(diffs != 0)[0], [ixn.size - 1]])

            iyn_N = iyn[changeIdx]

            sampledphase = iyn_N * (ft * Ts)
            sampledphase = sampledphase - sampledphase[0]

            if phasegram.size:
                sampledphase = sampledphase + phasegram[-1]

        curAmp = estAmp * np.ones(sampledphase.size)

        phasegram = np.concatenate([phasegram, sampledphase])
        amplitudegram = np.concatenate([amplitudegram, curAmp])

    if phasegram.size < 2:
        raise ValueError('JADE_v3:InsufficientPhaseData Insufficient phase data '
                          'were generated.')

    # Cubic spline interpolation at breakpoints
    if bpi > 1:
        breakpoints = np.arange(1, phasegram.size + 1, bpi)

        if breakpoints[-1] != phasegram.size:
            breakpoints = np.append(breakpoints, phasegram.size)
    else:
        breakpoints = zero_crossings[:-1] - (zero_crossings[0] - 1)

        breakpoints = breakpoints[(breakpoints >= 1) & (breakpoints <= phasegram.size)]

        breakpoints = _unique_stable(breakpoints)

    if breakpoints.size < 2:
        raise ValueError('JADE_v3:InsufficientBreakpoints At least two '
                          'interpolation breakpoints are required.')

    phase_at_bp = phasegram[breakpoints - 1]

    phasecurve = CubicSpline(breakpoints, phase_at_bp, bc_type='not-a-knot')

    phasevals = phasecurve(np.arange(1, breakpoints[-1] + 1))

    # Instantaneous frequency
    IFvals = np.diff(phasevals) / Ts

    if IF_N > 0:
        if IF_N >= IFvals.size:
            raise ValueError('JADE_v3:InvalidIFOrder IF_N must be smaller than '
                              'the number of IF samples.')

        print('smoothing IF estimate by polynomial fitting')

        total_x = np.arange(1, IFvals.size + 1)
        IFpoly = np.polyfit(total_x, IFvals, IF_N)
        IFvals = np.polyval(IFpoly, total_x)

    # Match amplitudegram length to phase output
    amplitudegram = amplitudegram[:min(amplitudegram.size, phasevals.size)]

    return IFvals, phasevals, zero_crossings, amplitudegram


def _dtw_cost(amp, template, cur_cycle):
    return _dtw_distance(amp * template, cur_cycle)


def _polyfit_zero(x, y, degree):
    """Port of polyfitZero.m (found on MATLAB File Exchange #35401):
    least-squares polynomial fit forced through the origin."""

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    dim = x.size

    if not (np.isfinite(degree) and 0 < degree <= 10 and degree == round(degree)):
        raise ValueError('polyfitZero:degreeOutOfRange DEGREE must be an integer '
                          'between 1 and 10.')

    if not (degree < dim):
        raise ValueError('polyfitZero:DegreeGreaterThanDim DEGREE must be less '
                          'than numel(X)')

    if x.size != y.size:
        raise ValueError('polyfitZero:vectorMismatch X and Y must be vectors of '
                          'the same length.')

    z = np.zeros((dim, degree))
    for n in range(1, degree + 1):
        z[:, n - 1] = x ** (degree - n + 1)

    p, _, _, _ = np.linalg.lstsq(z, y, rcond=None)
    p = np.concatenate([p, [0.0]])

    return p


def _scale_amplitudes(signal):
    """Port of scaleAmplitudes.m: iterative spline-envelope AM removal."""

    signal = np.asarray(signal, dtype=float).ravel()
    signal_scaled = signal.copy()
    correction_curve = np.ones_like(signal)

    for _ in range(3):
        tmp_correction_curve = _get_correction_curve(signal_scaled)
        signal_scaled = signal_scaled / tmp_correction_curve
        correction_curve = tmp_correction_curve * correction_curve

    instantaneous_amplitude = correction_curve

    return signal_scaled, instantaneous_amplitude


def _get_correction_curve(signal):
    """Port of getCorrectiongCurve.m."""

    signal = np.asarray(signal, dtype=float).ravel()
    signal_length = signal.size

    locs, _ = find_peaks(np.abs(signal))
    x_max = (locs + 1).astype(float)  # 1-based, matches MATLAB findpeaks locs
    y_max = np.abs(signal)[locs]

    x_max, y_max = _linear_spline_near_boundary(x_max, y_max, signal)

    correction_curve = CubicSpline(x_max, y_max, bc_type='not-a-knot')(
        np.arange(1, signal_length + 1))

    return correction_curve


def _linear_spline_near_boundary(x_max, y_max, signal):
    """Reconstruction of linearSplineNearBoundary.m (source not found in the
    repository). Ensures the peak list spans the full signal by linearly
    extrapolating the correction curve out to the first/last sample from the
    nearest two detected peaks, so the spline used in _get_correction_curve
    does not extrapolate wildly near the boundaries."""

    n = signal.size
    x = list(x_max)
    y = list(y_max)

    if len(x) == 0:
        level = np.max(np.abs(signal)) if n else 0.0
        return np.array([1.0, float(n)]), np.array([level, level])

    if x[0] != 1:
        if len(x) >= 2:
            slope = (y[1] - y[0]) / (x[1] - x[0])
            y0 = y[0] - slope * (x[0] - 1)
        else:
            y0 = y[0]
        x = [1.0] + x
        y = [y0] + y

    if x[-1] != n:
        if len(x) >= 2:
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            yend = y[-1] + slope * (n - x[-1])
        else:
            yend = y[-1]
        x = x + [float(n)]
        y = y + [yend]

    return np.array(x), np.array(y)


def _smooth_data(x):
    """Approximation of MATLAB smoothdata(x) with automatic 'movmean' window
    selection (exact window-selection heuristic is undocumented)."""

    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    window = max(3, int(round(np.sqrt(n))))
    if window % 2 == 0:
        window += 1

    half = window // 2
    padded = np.pad(x, (half, half), mode='edge')
    kernel = np.ones(window) / window

    return np.convolve(padded, kernel, mode='valid')


def _dtw(x, y):
    """Port of MATLAB's dtw(x, y) (default 'euclidean' metric, which reduces
    to absolute difference for scalar samples). Returns (dist, ix, iy) where
    ix/iy are 1-based warping-path indices into x and y respectively,
    matching MATLAB's [dist, ix, iy] = dtw(x, y)."""

    return _dtw_core(x, y, need_path=True)


def _dtw_distance(x, y):
    return _dtw_core(x, y, need_path=False)[0]


def _dtw_core(x, y, need_path):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n, m = x.size, y.size

    if n == 0 or m == 0:
        raise ValueError('dtw:emptyInput x and y must be nonempty.')

    cost = np.abs(x[:, None] - y[None, :])

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    # D[i,j] depends only on D[i-1,j], D[i,j-1], D[i-1,j-1] -- all on
    # anti-diagonals i+j-1 and i+j-2. Cells on the same anti-diagonal i+j=d
    # are therefore independent of each other, so each diagonal can be
    # filled with a single vectorized numpy step instead of a python-level
    # double loop. This is the exact same recurrence as the straightforward
    # nested loop (just reordered), so it produces bit-identical results,
    # only much faster.
    for d in range(2, n + m + 1):
        i_vals = np.arange(max(1, d - m), min(n, d - 1) + 1)
        if i_vals.size == 0:
            continue
        j_vals = d - i_vals

        step_cost = cost[i_vals - 1, j_vals - 1]
        D[i_vals, j_vals] = step_cost + np.minimum(
            np.minimum(D[i_vals - 1, j_vals], D[i_vals, j_vals - 1]),
            D[i_vals - 1, j_vals - 1],
        )

    dist = D[n, m]

    if not need_path:
        return dist, None, None

    ix_path = []
    iy_path = []
    i, j = n, m
    while i > 0 or j > 0:
        ix_path.append(i)
        iy_path.append(j)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            diag, up, left = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
            best = min(diag, up, left)
            if diag == best:
                i -= 1
                j -= 1
            elif up == best:
                i -= 1
            else:
                j -= 1

    ix_path.reverse()
    iy_path.reverse()

    return dist, np.array(ix_path), np.array(iy_path)


def _mround(v):
    """MATLAB-style round-half-away-from-zero (numpy/Python round to even)."""
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def _unique_stable(arr):
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


def _validate_nonneg_int(v, name):
    if not (np.isfinite(v) and v >= 0 and v == _mround(v)):
        raise ValueError(f'JADE_v3:InvalidArgument {name} must be a nonnegative integer.')
    return int(v)


def _validate_binary(v, name):
    if v not in (0, 1):
        raise ValueError(f'JADE_v3:InvalidArgument {name} must be 0 or 1.')
    return int(v)
