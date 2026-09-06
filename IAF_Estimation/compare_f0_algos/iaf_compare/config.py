"""Sweep configuration for the peak-frequency algorithm comparison, plus the
LaTeX conditions-table writer.

Signal generation uses the FOOOF generative model (see :mod:`iaf_compare.signal_gen`):
  "constant" stationarity: log10(power) = aperiodic(f) + peak(f) + noise(f),
    via fooof.sim.gen, synthesized into a real signal by random-phase irfft.
  "bursty" stationarity: genuine time-domain amplitude modulation, no
    log-power equivalent, synthesized directly and added on top of a
    separately generated aperiodic+noise background.

  aperiodic_ref_power_db : PSD of the aperiodic component at f_rotation, in dB
      (10*log10(power)). Global power anchor. 0 dB == 1 uV^2/Hz.
  f_rotation : spectral pivot frequency for the aperiodic shape; the aperiodic
      PSD equals aperiodic_ref_power_db exactly here.
  peak_snr_db : peak power relative to the aperiodic PSD at the peak's own
      peak_freq (FOOOF's own peak-height convention).
  noise_lv : std dev of the per-bin noise added in log10-power space.
"""
import numpy as np

# --- Sweep definition --------------------------------------------------------

FIXED = {
    "fs":                   10000.0,
    "freq_range":           (0.1, 30.0),
    "alpha_band":           (5, 18),
    "pink_ax_r2":           0.9,
    "aperiodic_ref_power_db": 0.0,  # anchor power at f_rotation, dB (0 dB == 1 uV^2/Hz)
    "f_rotation":           1.0,    # pivot freq so aperiodic shape is independent of carrier
}

DEFAULT = {
    "peak_freq":            np.float64(10.32),
    "stationarity":         "constant",   # "constant" | "bursty"
    "aperiodic_exponent":   2.0,          # β — slope of 1/f^β
    "n_peaks":              1,
    "peak_width":           0.5,          # Gaussian σ in Hz
    "peak_snr_db":          10.0,         # peak power relative to aperiodic floor at peak_freq
    "window_length_sec":    10,
    "noise_lv":             0.5,          # std dev of per-bin log10-power noise (gen_noise)
}

# window_length_sec is evaluated separately (crossed against every condition
# here -- see iaf_compare.pipeline.build_sweep), not as an OFAT sweep entry.
SWEEPS = {
    "peak_snr_db":          [0, 5, 10, 20, 50],
    "peak_width":           [0.1, 0.5, 1.0, 2.0],
    "aperiodic_exponent":   [0, 1, 2, 3],
    "peak_freq":            sorted([6, 8, 12, 14] + [DEFAULT["peak_freq"]]),
    "stationarity":         ["constant", "bursty"],
    "noise_lv":             [0, 0.2, 0.5, 1.0],
}

WINDOW_LENGTHS_SEC = [5, 10, 20]

N_SEEDS = 20  # number of noise variations for each condition 


# --- LaTeX conditions table ------------------------------------------------

name_dict = {
    "fs": "Sampling frequency [Hz]",
    "signal_length_sec": "Signal length [s]",
    "freq_range": "Frequency range [Hz]",
    "alpha_band": "Alpha band [Hz]",
    "peak_freq": "Fundamental frequency [Hz]",
    "stationarity": "Stationarity",
    "aperiodic_exponent": "Aperiodic exponent",
    "peak_width": "Peak width [Hz]",
    "peak_snr_db": "Peak SNR [dB]",
    "window_length_sec": "Window length [s]",
    "noise_lv": "Noise level ",  # std dev of per-bin noise in log10-power space
}

# Fixed params still worth reporting alongside the sweeps
_LATEX_TABLE_FIXED_PARAMS = ("fs", "freq_range", "alpha_band")


def _latex_escape(text):
    return str(text).replace("_", r"\_")


def _latex_format_value(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, tuple):
        return "--".join(_latex_format_value(v) for v in value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:g}"
    return _latex_escape(value)


def save_conditions_latex_table(default, sweeps, out_path, fixed=None,
                                fixed_params=_LATEX_TABLE_FIXED_PARAMS):
    """Write a LaTeX table: one row per swept parameter listing its tested
    values (default in bold), plus one row per fixed param."""

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Simulation parameters}",
        r"\label{tab:iaf_conditions}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        "Parameter & Tested values \\\\",
        r"\midrule",
    ]

    if fixed:
        for param in fixed_params:
            if param in fixed:
                param_name = name_dict.get(param, param)
                lines.append(rf"{_latex_escape(param_name)} & \textbf{{ {_latex_format_value(fixed[param])} }} \\")

    rows = [(param, values) for param, values in sweeps.items()]
    for param, values in rows:
        default_val = default.get(param)
        value_texts = []
        for val in values:
            text = _latex_format_value(val)
            if default_val is not None and val == default_val:
                text = rf"\textbf{{{text}}}"
            value_texts.append(text)
        lines.append(f"{_latex_escape(name_dict.get(param, param))} & {', '.join(value_texts)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_path.write_text("\n".join(lines) + "\n")
