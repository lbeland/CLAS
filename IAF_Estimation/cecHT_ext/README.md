# cecHT_ext — CLAS overlay on the `cecHT` submodule

[`../cecHT`](../cecHT) is a git submodule pinned to upstream
[`github.com/eikeosmers/cecHT`](https://github.com/eikeosmers/cecHT) and kept
**byte-identical** to it. All CLAS-specific work lives here and imports the shared
code (`phase`, `phase_track`, `utils`, `EEG/helpers`) from the pristine submodule
via [`_bootstrap.py`](_bootstrap.py).

## Layout

| path | what |
|------|------|
| `_bootstrap.py` | puts `../cecHT` and `../cecHT/EEG` on `sys.path` |
| `echt_ext.py` | `ECHTExt(phase.ECHT)` — custom `b`/`a` band-pass (`filter_type="custom"`) + sliding-DFT forward transform (`transform_sdft`) |
| `filtertestCecHT.py` | `ZeroPhaseButterECHT` — causal vs. zero-phase (`|H(f)|²`) band-pass comparison on a synthetic cosine |
| `eeg/helpers_ext.py` | per-window IAF, `load_ds004148`, `time_s` in the IAF CSV (ecHT phase-error part currently commented out) |
| `eeg/simple_iaf.py` | `combine_simple` (+ helpers) copied verbatim from `IAF_Estimation/IAF_tests.py`; `simple_paf` adapter |
| `eeg/run_pipeline.py` | driver: `ds004148`, `--iaf-method {fooof,simple}`, writes `iaf_per_segment.csv` |
| `eeg/plot_results.py` | IAF variability plots: one run dir (SD hist, std-vs-mean, SD-vs-mean-SNR) or two (fooof-vs-simple comparison) |
| `simulations/harmonic_experiments_sdft.py` | c-ecHT batch-FFT vs sliding-DFT sweep figure |
| `figures/` | generated figures (tracked) |
| `results/` | pipeline outputs — **git-ignored** (see `../../.gitignore`), kept on disk |

## Setup

```bash
git submodule update --init IAF_Estimation/cecHT
pip install -r IAF_Estimation/cecHT_ext/requirements.txt
```

## Run

Every entry point is a plain script (no `-m` needed); it self-bootstraps the
submodule path from `__file__`.

```bash
# Per-window IAF pipeline (ds004148, channel Fz, fooof estimator).
# Output -> cecHT_ext/results/<dataset>_<channel>_<method>/iaf_per_segment.csv
#           (here: results/ds004148_Fz_fooof/)
python IAF_Estimation/cecHT_ext/eeg/run_pipeline.py --dataset ds004148 --edf-dir /path/to/ds004148 --channel Fz --iaf-method fooof
#   --iaf-method {fooof,simple}   'simple' = combine_simple from IAF_tests.py
#   --out-dir DIR                 override the destination

# IAF variability (per-participant SD, Hz) — one run dir:
python IAF_Estimation/cecHT_ext/eeg/plot_results.py IAF_Estimation/cecHT_ext/results/ds004148_Fz_fooof
# ...or compare two methods (writes a Bland–Altman + SD comparison):
python IAF_Estimation/cecHT_ext/eeg/plot_results.py \
    IAF_Estimation/cecHT_ext/results/ds004148_Fz_fooof \
    IAF_Estimation/cecHT_ext/results/ds004148_Fz_simple
#   ^ iaf_stats_per_file.csv into the run dir(s); figures (.pgf + .pdf) into MA/plots/

# ECHT filter comparison
python IAF_Estimation/cecHT_ext/filtertestCecHT.py

# batch-FFT vs sliding-DFT sweep
python IAF_Estimation/cecHT_ext/simulations/harmonic_experiments_sdft.py
```

## Keeping the submodule pristine

To move to a newer upstream:

```bash
cd IAF_Estimation/cecHT && git fetch origin && git checkout origin/main
cd - && git add IAF_Estimation/cecHT && git commit -m "bump cecHT submodule"
```
