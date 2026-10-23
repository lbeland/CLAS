# research — offline studies behind CLAS

Offline research on the two quantities CLAS estimates in real time: the individual alpha
frequency (IAF), tracked by the `FrequencyEstimation` processor, and the instantaneous phase,
estimated by `PhaseEstimation`. Nothing in here is needed to run CLAS.

## Contents

| path | what |
|------|------|
| [`frequency/`](frequency/) | benchmark of IAF estimators on synthetic FOOOF-model signals |
| [`phase/`](phase/) | phase-estimation experiments and an automated SNR / noise sweep of the CLAS graphs |
| [`cecHT/`](cecHT/) | git submodule: upstream [cecHT](https://github.com/eikeosmers/cecHT) (calibrated end-corrected Hilbert transform), kept unmodified |
| [`cecHT_ext/`](cecHT_ext/) | CLAS additions on top of `cecHT`: IAF variability on real EEG (ds004148). See [cecHT_ext/README.md](cecHT_ext/README.md) |
| [`requirements.txt`](requirements.txt) | extra packages needed by the research scripts, on top of the main CLAS requirements |

### frequency/

Compares five IAF estimators: `Maximum`, `FOOOF`, `RestingIAF`, `alpha_fast` and
`alpha_fast_mt`. `alpha_fast` follows the same approach as CLAS's `FrequencyEstimation`
processor: aperiodic removal, Savitzky–Golay smoothing, a Gaussian peak fit and a BIC test.
The test signals are synthetic FOOOF-model spectra (`gen_power_vals_fn` in
`iaf_compare/signal_gen.py` is copied from
[TomDonoghue/SIMparam](https://github.com/TomDonoghue/SIMparam), MIT License).

| path | what |
|------|------|
| `iaf_compare/` | shared package: sweep config, signal generation, spectra, algorithms, metrics, HDF5 I/O, plot style |
| `run_sweep.py` | runs the full parameter sweep and writes `outputs/iaf_results.h5` |
| `plot_sweep.py` | figures and LaTeX tables from `iaf_results.h5` |
| `reject_compare.py` | compares peak-rejection (`approve_peak`) strategies for `alpha_fast`; writes `outputs/reject/strategy_comparison_raw.csv` |
| `approve_peak_variants.py` | the rejection strategies compared by `reject_compare.py` |
| `plot_reject.py` | scores the `reject_compare.py` output: summary CSV, LaTeX table, ROC plot |

**Outputs:**
- **Numeric results** stay local in `frequency/outputs/`.
- **Figures and `.tex` tables** go to `<repo>/analysis_output/outputs/` (see `iaf_compare/paths.py`).

### phase/

| path | what |
|------|------|
| `sweep_clas.py` | automated SNR / noise-colour sweep: runs a phase-estimation graph head-less through Falcon for every `(snr_db, noise_color)` point and collects the runs |
| `harmonic_experiments_offline_ht.py` | ecHT vs. c-ecHT vs. offline Hilbert transform on noisy cosines, across SNR; writes `figures/` |
| `phase_test.py` | does spectral whitening (removing the 1/f component) help or hurt bandpass + Hilbert phase estimation of an alpha tone in pink noise? |
| `hilbert_calib_demo.py` | phase error on a pure cosine: plain Hilbert vs. bandpass + Hilbert vs. ecHT with per-sample MSE-optimal calibration gains |

## Setup

The research scripts run in the main CLAS environment (see the main README). Install the
extra packages they need on top of it, and fetch the `cecHT` submodule:

```bash
pip install -r research/requirements.txt
git submodule update --init research/cecHT
```

## Run

Run everything from the repo root.

```bash
# IAF estimator sweep (slow), then figures/tables
python research/frequency/run_sweep.py
python research/frequency/plot_sweep.py

# Peak-rejection strategy comparison (slow), then scoring/plots
python research/frequency/reject_compare.py
python research/frequency/plot_reject.py

# ecHT vs. c-ecHT vs. offline Hilbert sweep -> phase/figures/
python research/phase/harmonic_experiments_offline_ht.py

# Phase-estimation demos, save a .png in the current directory
python research/phase/phase_test.py
python research/phase/hilbert_calib_demo.py
```

### sweep_clas.py

Needs a built Falcon (`build/release`) and a graph that uses `SimulatedSource` (default:
[`ecHTtests.yaml`](../resources/graphs/ecHTtests.yaml)):

```bash
python research/phase/sweep_clas.py --dry-run      # print the plan and exit
python research/phase/sweep_clas.py
python research/phase/sweep_clas.py --graphs ecHTtests.yaml --snr -10 0 10 20 50 --noise white pink
```

**How a sweep runs:**
- **Results:** runs are collected in `results/<name>/<graph>/`; `--name` defaults to `snr_sweep`.
- **Summary:** at the end, one phase-error table per graph is written via
  [`analysis/sweep.py`](../analysis/sweep.py). Use `--no-analyse` to skip it.
- **Your graph is not modified:** the sweep writes a temporary copy (`_sweep_<graph>`) with the
  SNR and noise settings patched in, and deletes it afterwards.

For `cecHT_ext/`, see [cecHT_ext/README.md](cecHT_ext/README.md).
