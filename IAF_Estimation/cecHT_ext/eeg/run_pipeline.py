"""Per-window individual-alpha-frequency (IAF) pipeline across EEG datasets (CLAS overlay).

Based on upstream ``EEG/eeg_phase.py`` but:
- adds the ``ds004148`` dataset,
- ``--iaf-method {fooof,alpha_fast}`` selects the per-window PAF estimator
  (``alpha_fast`` = the ``alpha_fast`` algorithm from ``analysis/f0.py``),
- shows a progress bar and pins worker CPU affinity,
- ``--subjects`` is honoured by every loader.

The ecHT phase-error analysis is currently disabled in :mod:`helpers_ext`
(``process_segment`` / ``aggregate_and_save``); only the IAF series is produced.

Output (default: ``cecHT_ext/results/<dataset>_<channel>_<method>/``; override
with ``--out-dir``):
- iaf_per_segment.csv   per-window IAF estimates (file, segment_index, had_alpha,
                        paf_hz, time_s)

Then plot / compare with ``eeg/plot_results.py``.

Example
-------
    python <cecHT_ext>/eeg/run_pipeline.py --dataset ds004148 \
        --edf-dir /path/to/ds004148 --channel Fz-FCz --iaf-method fooof
    # -> writes cecHT_ext/results/ds004148_Fz-FCz_fooof/iaf_per_segment.csv
"""

import os
import re
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401  -> puts the cecHT submodule on sys.path

from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

try:
    from tqdm_joblib import tqdm_joblib  # noqa: E402
except ImportError:  # pragma: no cover - fall back to no progress bar
    import contextlib

    @contextlib.contextmanager
    def tqdm_joblib(_tqdm):
        yield

import helpers_ext as H  # noqa: E402

_RESULTS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "results"
_IAF_NAME = "iaf_per_segment.csv"
# phase-error output names (analysis disabled; kept for when it is re-enabled)
_CSV_NAME = "phase_error_per_file.csv"
_NPZ_NAME = "phase_error_all.npz"
_N_JOBS = -1

# Effective channel when ``--channel`` is not given (mirrors each loader's default).
_DEFAULT_CHANNEL = {
    "ds004148": "Fz-FCz",
    "hmc": "EEG O2-M1",
    "rodrigues2017": "Oz",
}


def _slug(text):
    """Filesystem-safe token (spaces/slashes/... -> '-')."""
    return re.sub(r"[^\w.+-]+", "-", str(text)).strip("-")


def default_out_dir(dataset, channel_name, iaf_method):
    """cecHT_ext/results/<dataset>_<channel>_<method>/."""
    return _RESULTS_ROOT / f"{_slug(dataset)}_{_slug(channel_name)}_{_slug(iaf_method)}"


def set_affinity():
    """Reset CPU affinity in loky workers (joblib can pin them to one core)."""
    try:
        os.sched_setaffinity(0, set(range(os.cpu_count())))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux / isolated cores
        pass


def main(
    dataset,
    iaf_window=10.0,
    max_subjects=None,
    conditions=None,
    bw_factor=0.5,
    filt_order=1,
    edf_dir=None,
    channel_name=None,
    iaf_method="fooof",
    out_dir=None,
):
    if dataset not in _DEFAULT_CHANNEL:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    # Resolve the effective channel so the output dir always reflects it.
    channel = channel_name or _DEFAULT_CHANNEL[dataset]

    out_dir = pathlib.Path(
        out_dir if out_dir is not None
        else default_out_dir(dataset, channel, iaf_method)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    iaf_csv_path = str(out_dir / _IAF_NAME)
    csv_path = str(out_dir / _CSV_NAME)   # unused (phase-error analysis disabled)
    npz_path = str(out_dir / _NPZ_NAME)   # unused (phase-error analysis disabled)
    print(f"Output dir: {out_dir}")

    if dataset == "hmc":
        if edf_dir is None:
            raise ValueError("--edf-dir is required for the HMC dataset")
        segments = H.load_hmc(edf_dir=edf_dir, channel_name=channel)

    elif dataset == "rodrigues2017":
        segments = H.load_rodrigues2017(conditions=conditions, channel_name=[channel])

    elif dataset == "ds004148":
        if edf_dir is None:
            raise ValueError("--edf-dir is required for the ds004148 dataset")
        segments = H.load_ds004148(edf_dir=edf_dir, max_subjects=max_subjects,
                                   channel_name=channel)

    if not segments:
        print("No segments loaded. Exiting.")
        return

    print(f"\n{len(segments)} segments  |  "
          f"IAF: {iaf_method} / {iaf_window:.1f} s  |  "
          f"bw_factor: {bw_factor}  |  filt_order: {filt_order}")

    with tqdm_joblib(tqdm(total=len(segments))):
        results = Parallel(n_jobs=_N_JOBS, backend="loky", initializer=set_affinity)(
            delayed(H.process_segment)(seg, iaf_window, bw_factor, filt_order, iaf_method)
            for seg in segments
        )
    for r in results:
        print(f"  {r['seg_id']}: {r['reason'] or 'OK'}")

    H.aggregate_and_save(results, csv_path, npz_path, iaf_csv_path)
    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compute ecHT phase-estimation errors for EEG datasets.")

    p.add_argument("--dataset", choices=["hmc", "rodrigues2017", "ds004148"],
                   default="rodrigues2017")
    p.add_argument("--iaf-window", type=float, default=10,
                   help="IAF estimation window (s). <=0 -> full segment. (default: 10)")
    p.add_argument("--iaf-method", choices=["fooof", "alpha_fast"], default="alpha_fast",
                   help="Per-window IAF estimator: 'fooof' (upstream) or 'alpha_fast' "
                        "(the alpha_fast algorithm from analysis/f0.py). (default: alpha_fast)")
    p.add_argument("--subjects", type=int, default=None,
                   help="Max number of subjects, ds004148 only (default: all).")
    p.add_argument("--conditions", nargs="*", default="EC",
                   help="Conditions to include, e.g. EC EO (Rodrigues2017 only).")
    p.add_argument("--bw", type=float, default=0.5,
                   help="Bandwidth factor: f0 ± bw*f0/2 (default: 0.5).")
    p.add_argument("--filt-order", type=int, default=1,
                   help="Butterworth filter order (default: 1).")
    p.add_argument("--edf-dir", type=str, default=None,
                   help="Directory with EDF files (required for hmc / ds004148).")
    p.add_argument("--channel", type=str, default=None,
                   help="EEG channel name (e.g. Fz, Pz; default: loader-specific).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Directory for the 3 output files "
                        "(default: cecHT_ext/results/<dataset>_<channel>_<method>/).")

    args = p.parse_args()
    main(
        dataset=args.dataset,
        iaf_window=args.iaf_window,
        max_subjects=args.subjects,
        conditions=args.conditions,
        bw_factor=args.bw,
        filt_order=args.filt_order,
        edf_dir=args.edf_dir,
        channel_name=args.channel,
        iaf_method=args.iaf_method,
        out_dir=args.out_dir,
    )
