# cecHT_ext — CLAS overlay on the `cecHT` submodule

[`../cecHT`](../cecHT) is a git submodule pinned to upstream
[`github.com/eikeosmers/cecHT`](https://github.com/eikeosmers/cecHT) and kept
**byte-identical** to it. All CLAS-specific work lives here and imports the shared
code (`phase`, `phase_track`, `utils`, `EEG/helpers`) from the pristine submodule
via [`_bootstrap.py`](_bootstrap.py).

## Layout

| path | what |
|------|------|
| `_bootstrap.py` | puts `../cecHT` and `../cecHT/EEG` on `sys.path` (also used by `../phase/harmonic_experiments_offline_ht.py`) |
| `eeg/helpers_ext.py` | per-window IAF, `load_ds004148`, `time_s` in the IAF CSV (ecHT phase-error part currently commented out) |
| `eeg/alpha_fast_iaf.py` | `alpha_fast` (+ helpers) copied verbatim from `analysis/offline_estimation.py`; `alpha_fast_paf` adapter |
| `eeg/run_pipeline.py` | driver: `ds004148`, `--iaf-method {fooof,alpha_fast}`, writes `iaf_per_segment.csv` |
| `eeg/plot_results.py` | IAF variability plots: one run dir (SD hist, std-vs-mean, SD-vs-mean-SNR) or two (fooof-vs-alpha_fast comparison) |
| `results/` | pipeline outputs — **git-ignored**, kept on disk |

## Setup

```bash
git submodule update --init research/cecHT
pip install -r research/requirements.txt   # on top of the main CLAS requirements
```

## Run

Every entry point is a plain script (no `-m` needed); it self-bootstraps the
submodule path from `__file__`.

```bash
# Per-window IAF pipeline (ds004148, channel Fz, fooof estimator).
# Output -> cecHT_ext/results/<dataset>_<channel>_<method>/iaf_per_segment.csv
#           (here: results/ds004148_Fz_fooof/)
python research/cecHT_ext/eeg/run_pipeline.py --dataset ds004148 --edf-dir /path/to/ds004148 --channel Fz --iaf-method fooof
#   --iaf-method {fooof,alpha_fast}   'alpha_fast' = the alpha_fast algorithm from analysis/offline_estimation.py
#   --out-dir DIR                     override the destination

# IAF variability (per-participant SD, Hz) — one run dir:
python research/cecHT_ext/eeg/plot_results.py research/cecHT_ext/results/ds004148_Fz_fooof
# ...or compare two methods (writes a Bland–Altman + SD comparison):
python research/cecHT_ext/eeg/plot_results.py \
    research/cecHT_ext/results/ds004148_Fz_fooof \
    research/cecHT_ext/results/ds004148_Fz_alpha_fast
#   ^ iaf_stats_per_file.csv into the run dir(s); figures (.pgf + .pdf) into <repo>/analysis_output/
```

## Keeping the submodule pristine

To move to a newer upstream:

```bash
cd research/cecHT && git fetch origin && git checkout origin/main
cd - && git add research/cecHT && git commit -m "bump cecHT submodule"
```
