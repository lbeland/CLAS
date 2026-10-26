# Offline analysis

`clas.py` runs a full analysis automatically after a recording finishes, but this package can
also be run standalone against any `results/` folder. Everything runs through one CLI;
the rest of the package is a supporting library imported by it (and by `clas.py`).

**[`python -m analysis.main <command>`](main.py)** (from `CLAS/`), with commands:

- `single [results_dir] [--f0 HZ] [--f0-is-truth] [--whiten] [--no-plot] [--no-full-analysis]`
  -- full analysis + plots for one results folder (default: `_last_run`).
- `all [--base-dir DIR] [--f0 HZ] [--no-full-analysis]` -- analyse every results folder under
  `DIR` (default: `results`).
- `pooled [--f0 HZ] [--base-dir DIR] [--session-glob GLOB] [--names SUBSTR ...]` -- pool
  phase/f0/stimulus-edge errors across every recording in the session folders and plot/print
  the combined statistics ([`pooled.py`](pooled.py)).
- `erp [--base-dir DIR] [--session-glob GLOB] [--channel IDX]` -- plot one averaged
  ERP curve per subject ([`erp.py`](erp.py)).
- `spectrum [--base-dir DIR] [--channel NAME] [--win-s S] [--step-s S] [--smooth-window-s S]
  [--no-csd] [--stim-onset-degs DEG ...] [--comparisons A,B ...]` -- time-frequency /
  power-change analysis across stimulus-onset conditions (Stim On vs Stim Off), pooled across
  recordings ([`spectrum_analysis.py`](spectrum_analysis.py)).
- `sweep [sweep_dir] [--f0 HZ] [--no-f0-is-truth] [--whiten] [--out-dir DIR] [--label LABEL]`
  -- error table of an SNR/noise-colour sweep, as produced by
  [`research/phase/sweep_clas.py`](../research/phase/sweep_clas.py) ([`sweep.py`](sweep.py)).

Run `python -m analysis.main <command> --help` for the full flag list.
`python -m analysis.erp` additionally opens a folder picker to average the ERP of hand-picked runs.

The rest of the package:

- Reading and writing files:
  - [`clas_file_reader.py`](clas_file_reader.py) -- reads what a CLAS session writes (processor
    `.bin` output, stim protocol log, graph `.yaml`).
  - [`results_io.py`](results_io.py) -- reads/writes results files of analysis.
  - [`cache.py`](cache.py) -- `load_signals()`: the first analysis of a results folder parses
    the `.bin` files and writes `raw_signals.edf` + `runtime_metadata.h5` into it; later runs
    load those instead, which is much faster.
- Analysis:
  - [`pipeline.py`](pipeline.py) -- `load_and_analyse()`: the full offline analysis of one
    results folder, which every command builds on.
  - [`offline_estimation.py`](offline_estimation.py) -- offline f0 and Hilbert phase reference
    via `estimate_offline()`.
  - [`estimation_errors.py`](estimation_errors.py) -- phase/f0/stimulus-edge errors of the
    online estimates.
  - [`stimulus.py`](stimulus.py) -- stimulus reconstruction and the study's stimulation
    conditions.
  - [`latency.py`](latency.py) -- pipeline and audio onset latency.
  - [`erp.py`](erp.py) -- ERP epoching.
- Output:
  - [`plot.py`](plot.py) -- figures and LaTeX tables, saved to `CLAS/analysis_output/`.
  - [`report.py`](report.py) -- terminal report.

[`plot.py`](plot.py) renders all plot text through a real LaTeX installation
(`text.usetex`/`pgf.texsystem: pdflatex`), so a working `pdflatex` (e.g. `texlive`) needs to be
on `PATH`.
