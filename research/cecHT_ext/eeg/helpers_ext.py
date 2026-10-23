"""Local EEG-pipeline extensions layered on the pristine ``cecHT`` submodule.

Everything that is unchanged from upstream ``EEG/helpers.py`` is imported from
there. This module only redefines the pieces the CLAS work needed:

* per-window IAF with window-centre timestamps (``estimate_paf`` returns a tuple)
* window length ``round(2*fs/f0)`` instead of the fixed ``round(0.512*fs)``
* ``echt_vs_hilbert`` driven by :mod:`phase_track` with a per-sample ``f0_seq``
* an ``ds004148`` loader (``load_hmc`` / ``load_rodrigues2017`` are upstream)
* ``process_segment`` with a low-pass-only pre-filter and interpolated ``f0_seq``
* ``time_s`` carried through into ``iaf_per_segment.csv``
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401  -> puts the cecHT submodule on sys.path

import csv  # noqa: E402
import os  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import mne  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import interp1d  # noqa: E402
from scipy.signal import butter, hilbert, sosfiltfilt  # noqa: E402

# Upstream ``EEG/helpers.py`` does ``from fooof import FOOOF`` at import time, and
# fooof 1.1's __init__ calls ``simplefilter('always')`` and then ``warn(...)`` so
# its multi-line DeprecationWarning banner prints on *every* import regardless of
# the active filters -- once per loky worker process here. Suppress just that one
# message by dropping it at ``showwarning`` level (filters can't stop it), and use
# ``catch_warnings`` so fooof's global ``simplefilter('always')`` is reverted too.
with warnings.catch_warnings():
    _real_showwarning = warnings.showwarning

    def _drop_fooof_deprecation(message, category, *args, **kwargs):
        if issubclass(category, DeprecationWarning) and "fooof" in str(message):
            return
        return _real_showwarning(message, category, *args, **kwargs)

    warnings.showwarning = _drop_fooof_deprecation
    import helpers as _h  # noqa: E402  -> pristine upstream EEG/helpers.py
from phase_track import ECHT as ECHT_track  # noqa: E402
from utils import _circ_stats, run_echt_window_loop  # noqa: E402

from alpha_fast_iaf import alpha_fast_paf  # noqa: E402  -> alpha_fast, ported from analysis/f0.py

# Re-export upstream primitives that are reused unchanged.
iaf = _h.iaf
_fail = _h._fail
IafResult = _h.IafResult
get_first_stage_change_end = _h.get_first_stage_change_end
HMC_EXCLUDE_SUBJECTS = _h.HMC_EXCLUDE_SUBJECTS
load_hmc = _h.load_hmc
load_rodrigues2017 = _h.load_rodrigues2017


# --------------------------------------------------------------------------
# IAF estimation with per-window timestamps
# --------------------------------------------------------------------------
def estimate_paf(data, info, fmin=7.5, fmax=14,
                 segment_duration=6, polyorder=5, step=0.15, method="fooof"):
    """Median PAF across sliding IAF windows, plus the per-window PAFs/times.

    Parameters
    ----------
    method : {"fooof", "alpha_fast"}
        ``"fooof"``      -> upstream :func:`helpers.iaf` (FOOOF aperiodic + BIC test).
        ``"alpha_fast"`` -> :func:`alpha_fast_iaf.alpha_fast_paf` (log-log 1/f fit,
        Savgol-smoothed residual, BIC peak approval) -- the ``alpha_fast``
        algorithm from ``analysis/f0.py``.

    Returns
    -------
    (median_paf | None, pafs: list[float], times: list[float], snrs: list[float])
        ``times`` are window centres in seconds; ``snrs`` is per-window SNR for
        ``method="alpha_fast"`` (``nan`` for ``method="fooof"``).
    """
    if method not in ("fooof", "alpha_fast"):
        raise ValueError(f"unknown IAF method: {method!r}")

    fs = float(info["sfreq"])
    raw_tmp = mne.io.RawArray(data[np.newaxis, :], info)
    duration = raw_tmp.times[-1]
    segment_duration = min(duration, segment_duration)

    # window start times; when the segment is (about) one window long
    # `np.arange` is empty -- fall back to a single window covering all of it
    starts = np.arange(0, duration - segment_duration+1, step)
    if starts.size == 0:
        starts = np.array([0.0])

    pafs, times, snrs = [], [], []
    for tmin in starts:
        tmax = min(tmin + segment_duration, duration)
        seg = raw_tmp.copy().crop(tmin=tmin, tmax=tmax)
        n_win = seg.n_times
        try:
            if method == "fooof":
                # upstream iaf() uses n_fft = int(fs / resolution) and MNE's
                # Welch rejects n_fft > n_times; cap the resolution for short
                # windows (e.g. the ~10 s Rodrigues2017 blocks) so it stays <=.
                res_hz = max(0.1, fs / n_win * 1.05)
                r = iaf(seg, fmin=fmin, fmax=fmax, pink_max_r2=0.9, resolution=res_hz)
                paf, snr = r.PeakAlphaFrequency, np.nan
            else:
                paf, snr = alpha_fast_paf(seg.get_data()[0], fs, alpha_band=(fmin, fmax))
        except Exception:  # noqa: BLE001 - a single bad window must not kill the segment
            continue
        if paf is not None and np.isfinite(paf) and paf > 0:
            pafs.append(float(paf))
            times.append(tmin + segment_duration / 2)  # centre of window
            snrs.append(float(snr) if snr is not None else np.nan)

    if not pafs:
        return None, [], [], []
    return float(np.median(pafs)), pafs, times, snrs


def params_from_f0(fs, f0, bw_factor=0.5):
    """ecHT window length and band-pass edges from centre frequency.

    ``win_len`` spans ~2 cycles of ``f0`` (vs. upstream's fixed ``0.512*fs``).
    """
    win_len = int(round(2 * fs / f0))
    bw = bw_factor * f0
    l_freq = max(f0 - bw / 2, 0.1)
    h_freq = min(f0 + bw / 2, fs / 2 - 0.1)
    return win_len, l_freq, h_freq


def echt_vs_hilbert(data, fs, filt_order, f0, l_freq, h_freq, win_len,
                    ref_analytic_signal, f0_seq=None):
    """ecHT online (uncalibrated + calibrated) vs. the acausal reference (deg).

    Uses :class:`phase_track.ECHT`; ``f0_seq`` is a per-sample centre-frequency
    array (defaults to constant ``f0``).
    """
    n = data.size
    if win_len < 3 or win_len >= n:
        raise ValueError(f"invalid window length (win_len={win_len}, N={n})")

    echt_unc = ECHT_track(l_freq=l_freq, h_freq=h_freq, sfreq=fs,
                          filt_order=filt_order, calibrate=False)
    echt_cal = ECHT_track(l_freq=l_freq, h_freq=h_freq, sfreq=fs,
                          filt_order=filt_order, f0=f0, calibrate=True)
    echt_unc.fit(data[:win_len])
    echt_cal.fit(data[:win_len])

    ref_phase = np.angle(ref_analytic_signal)
    if f0_seq is None:
        f0_array = np.full_like(data, f0)
    else:
        assert f0_seq.shape == data.shape, "f0 array must match data shape"
        f0_array = f0_seq

    err_unc, err_cal = run_echt_window_loop(
        data, ref_phase, echt_unc, echt_cal, win_len, f0_seq=f0_array
    )
    return np.degrees(err_unc), np.degrees(err_cal)


# --------------------------------------------------------------------------
# Dataset loaders  (load_hmc / load_rodrigues2017 are upstream, re-exported above)
# --------------------------------------------------------------------------
def load_ds004148(edf_dir, max_subjects=None, channel_name="Fz-FCz"):
    """Load ds004148 eyes-closed resting-state EDFs into windowed segments."""
    edf_dir = Path(edf_dir)
    edf_files = sorted(edf_dir.glob("*task-eyesclosed_eeg.edf"))
    if max_subjects is not None:
        edf_files = edf_files[:max_subjects]

    segments = []
    for edf_path in edf_files:
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            if channel_name not in raw.ch_names:
                continue

            full_data = raw.pick([channel_name]).get_data().squeeze()
            fs = float(raw.info["sfreq"])
            full_info = raw.info.copy()

            total_dur = full_data.size / fs
            if total_dur > 300:
                window_dur, step_dur = 300, 150
            else:
                window_dur, step_dur = total_dur, 1

            current_t, block_idx = 0, 0
            while current_t + window_dur <= total_dur:
                s0 = int(round(current_t * fs))
                s1 = min(int(round((current_t + window_dur) * fs)), full_data.size)
                if (s1 - s0) < 10:
                    break
                segments.append(dict(
                    subject=edf_path.stem.replace("-", ""),
                    condition="EC",
                    block_idx=block_idx,
                    channel=channel_name,
                    duration=window_dur,
                    full_data=full_data,
                    full_info=full_info,
                    fs=fs,
                    sample_start=s0,
                    sample_end=s1,
                ))
                current_t += step_dur
                block_idx += 1
        except Exception as e:  # noqa: BLE001 - match upstream loader behaviour
            print(f"  Skipping {edf_path.name}: {e}")

    print(f"Loaded {len(segments)} windows from ds004148 ({edf_dir})")
    return segments


# --------------------------------------------------------------------------
# Per-segment processing
# --------------------------------------------------------------------------
def process_segment(seg, iaf_window=10, bw_factor=0.5, filt_order=1, iaf_method="fooof"):
    """Per-window IAF estimates for one segment.

    ``iaf_method`` ("fooof" | "alpha_fast") selects the per-window PAF estimator
    (see :func:`estimate_paf`).

    NOTE: the ecHT phase-error analysis is currently disabled -- only the IAF
    variability is needed. Re-enable the commented block below (and the matching
    block in :func:`aggregate_and_save`) to restore it.
    """
    mne.set_log_level("ERROR")
    sid = f"s{seg['subject']}_{seg.get('condition', '')}_b{seg.get('block_idx', 0)}"

    try:
        x = seg["full_data"][seg["sample_start"]:seg["sample_end"]].copy()
        fs = seg["fs"]
        info = seg["full_info"].copy()

        if x.size < 10:
            return _fail(sid, f"segment too short (N={x.size})")

        # Pre-filter: <=40 Hz low-pass + 46-54 Hz notch
        for sos in [butter(4, 40, fs=fs, btype="lowpass", output="sos"),
                    butter(4, [46, 54], fs=fs, btype="stop", output="sos")]:
            x = sosfiltfilt(sos, x)

        # step size of 1 seconds
        mean_paf, pafs, times, snrs = estimate_paf(
            x, info, segment_duration=iaf_window, method=iaf_method, step=1,
        )
        if mean_paf is None:
            return _fail(sid, "no_alpha: no valid PAF")

        # --- ecHT phase-error analysis (disabled: only IAF variability wanted) ---
        # f0_seq = interp1d(
        #     times, pafs, kind="nearest",
        #     bounds_error=False, fill_value="extrapolate",
        # )(np.arange(x.size) / fs)
        #
        # win_len, l_freq, h_freq = params_from_f0(fs, mean_paf, bw_factor)
        # if win_len < 3 or win_len >= x.size:
        #     return _fail(sid, f"invalid window length ({win_len=}, N={x.size})")
        #
        # # Acausal reference on the full recording, then slice
        # sos = butter(filt_order, [l_freq, h_freq], fs=fs, btype="band", output="sos")
        # ref = hilbert(sosfiltfilt(sos, seg["full_data"]))
        # ref_seg = ref[seg["sample_start"]:seg["sample_end"]]
        #
        # pe_unc, pe_cal = echt_vs_hilbert(
        #     x, fs, filt_order, mean_paf, l_freq, h_freq, win_len,
        #     ref_analytic_signal=ref_seg, f0_seq=f0_seq,
        # )
        pe_unc = pe_cal = None
        # -----------------------------------------------------------------------

        return dict(
            seg_id=sid, ok=True, reason="",
            phase_err_unc=pe_unc, phase_err_cal=pe_cal,
            iaf_segments=[
                dict(segment_index=i, had_alpha=1, paf_hz=p, time_s=t, snr=snr)
                for i, (p, t, snr) in enumerate(zip(pafs, times, snrs))
            ],
        )

    except Exception as e:  # noqa: BLE001 - match upstream behaviour
        return _fail(sid, f"error: {e}")


def aggregate_and_save(results, csv_path, npz_path, iaf_csv_path):
    """Write ``iaf_per_segment.csv`` (file, segment_index, had_alpha, paf_hz, time_s, snr).

    ``csv_path`` / ``npz_path`` are accepted for signature compatibility but are
    unused while the ecHT phase-error analysis is disabled (see the commented
    block below and in :func:`process_segment`).
    """
    iaf_rows = []
    no_alpha, errors = [], []
    n_valid = 0

    for r in results:
        if not r["ok"]:
            bucket = no_alpha if "no_alpha" in r["reason"] else errors
            bucket.append((r["seg_id"], r["reason"]))
            continue

        n_valid += 1
        iaf_rows.extend(
            dict(file=r["seg_id"], segment_index=s["segment_index"],
                 had_alpha=s["had_alpha"], paf_hz=s["paf_hz"],
                 time_s=s.get("time_s"), snr=s.get("snr"))
            for s in r.get("iaf_segments", [])
        )

    print(f"\n=== Summary ===\n"
          f"  Total: {len(results)}   Valid: {n_valid}   "
          f"No alpha: {len(no_alpha)}   Errors: {len(errors)}")
    for label, items in [("No alpha", no_alpha), ("Errors", errors)]:
        for path, reason in items:
            print(f"    [{label}] {path}: {reason}")

    os.makedirs(os.path.dirname(iaf_csv_path) or ".", exist_ok=True)
    with open(iaf_csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["file", "segment_index", "had_alpha", "paf_hz", "time_s", "snr"]
        )
        w.writeheader()
        w.writerows(iaf_rows)
    print(f"IAF estimates -> {iaf_csv_path}")

    # --- ecHT phase-error aggregation (disabled) ----------------------------
    # all_unc, all_cal, per_file_rows = [], [], []
    # for r in results:
    #     if not r["ok"]:
    #         continue
    #     pe_unc, pe_cal = r["phase_err_unc"], r["phase_err_cal"]
    #     all_unc.append(pe_unc)
    #     all_cal.append(pe_cal)
    #     m_u, s_u, plv_u, pli_u = _circ_stats(np.radians(pe_unc))
    #     m_c, s_c, plv_c, pli_c = _circ_stats(np.radians(pe_cal))
    #     per_file_rows.append(dict(
    #         file=r["seg_id"], n_samples=pe_unc.size,
    #         mean_unc_deg=np.degrees(m_u), std_unc_deg=np.degrees(s_u),
    #         plv_unc=plv_u, pli_unc=pli_u,
    #         mean_cal_deg=np.degrees(m_c), std_cal_deg=np.degrees(s_c),
    #         plv_cal=plv_c, pli_cal=pli_c,
    #     ))
    # fields = ["file", "n_samples",
    #           "mean_unc_deg", "std_unc_deg", "plv_unc", "pli_unc",
    #           "mean_cal_deg", "std_cal_deg", "plv_cal", "pli_cal"]
    # with open(csv_path, "w", newline="") as f:
    #     w = csv.DictWriter(f, fieldnames=fields)
    #     w.writeheader()
    #     w.writerows(per_file_rows)
    # np.savez(npz_path,
    #          phase_err_unc_deg_all=np.concatenate(all_unc),
    #          phase_err_cal_deg_all=np.concatenate(all_cal))
    # ---------------------------------------------------------------------
