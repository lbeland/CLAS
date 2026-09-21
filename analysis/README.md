# Offline analysis

`clas.py` runs a full analysis automatically after a recording finishes, but this package can
also be run standalone against any `results/` folder. It has two CLI entry points; everything
else in the package is a supporting library imported by these (and by `clas.py`) rather than
something you run directly.

- **[`python -m analysis.main <command>`](main.py)** -- the main CLI, with subcommands:
  - `single [results_dir] [--f0 HZ] [--f0-is-truth] [--whiten] [--no-show] [--no-full-analysis]`
    -- full analysis + plots for one results folder (default: `_last_run`).
  - `all [--base-dir DIR] [--f0 HZ] [--no-full-analysis]` -- analyse every results folder under
    `DIR` (default: `results`).
  - `pooled [--f0 HZ] [--base-dir DIR] [--session-glob GLOB] [--names SUBSTR ...]` -- pool
    phase/f0/stimulus-edge errors across every recording in a session and plot/print the
    combined statistics.
  - `erp [--f0 HZ] [--base-dir DIR] [--session-glob GLOB] [--channel IDX]` -- plot one averaged
    ERP curve per subject.
  - Run `python -m analysis.main <command> --help` for the full flag list.
- **[`python -m analysis.spectrum_analysis`](spectrum_analysis.py) [--base-dir DIR]
  [--channel NAME] [--win-s S] [--step-s S] [--smooth-window-s S] [--no-csd]
  [--stim-onset-degs DEG ...] [--comparisons A,B ...]** -- time-frequency / power-change
  analysis across stimulus-onset conditions (Stim On vs Stim Off), pooled across recordings.
  Run with `--help` for the full flag list.

[`sweep.py`](sweep.py) (SNR/noise-colour sweep tables, as produced by
[`sweep_clas.py`](../sweep_clas.py)) has no CLI yet -- edit the call in its
`if __name__ == "__main__":` block, or import and call `snr_sweep_table()` yourself.

The rest of the package ([`core.py`](core.py) Hilbert reference/error computation,
[`f0.py`](f0.py) f0 estimation, [`modal.py`](modal.py) the MODAL algorithm,
[`loader.py`](loader.py) raw `.bin` loading,
[`edf_io.py`](edf_io.py)/[`runtime_meta.py`](runtime_meta.py) EDF+HDF5
caching, [`plot.py`](plot.py) plotting/tables,
[`stimulus.py`](stimulus.py) stimulus reconstruction) is imported by the entry points
above. Notably, the first analysis of a results folder writes `raw_signals.edf` +
`runtime_metadata.h5` into it; later runs against the same folder load those instead of
re-parsing the raw `.bin` files, which is much faster.

Two things to be aware of before running any of this:
- [`plot.py`](plot.py) renders all plot text through a real LaTeX installation
  (`text.usetex`/`pgf.texsystem: pdflatex`), so a working `pdflatex` (e.g. `texlive`) needs to be
  on `PATH`.
- [`plot.py`](plot.py)'s `PLOTS_DIR` is currently a hardcoded absolute path -- edit it to your own
  output directory before running anything that saves a figure or table.
