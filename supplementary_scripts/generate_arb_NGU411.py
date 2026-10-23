#!/usr/bin/env python3
"""
NGU411 Rohde & Schwarz Arbitrary Waveform File Generator
Generates sine wave .csv files compatible with the NGU411 ARB input format.
"""

import math
import argparse
from fractions import Fraction

# NGU411 ARB hardware limits.
NGU411_MAX_POINTS = 4096
NGU411_MIN_DWELL_S = 1e-4       # minimum dwell time between two ARB points: 0.1 ms
NGU411_DURATION_DECIMALS = 4    # step-duration resolution: 0.1 ms


def _divisors(n: int):
    """All positive divisors of n, ascending."""
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs)


def plan_arb(f: Fraction):
    """
    Plan an ARB block that reproduces `f` with no rounding in the step duration.

    With ``f = p / q`` and ``scale = 10**NGU411_DURATION_DECIMALS`` ticks per
    second, ``p / gcd(p, scale)`` whole cycles take exactly
    ``q * scale / gcd(p, scale)`` ticks, which split into equal, exact steps.
    Rep loops that block.

    Returns (num_cycles, num_points, duration_per_point_s).
    """
    if f <= 0:
        raise ValueError("frequency must be positive")

    scale = 10 ** NGU411_DURATION_DECIMALS
    p, q = f.numerator, f.denominator
    g = math.gcd(p, scale)
    num_cycles = p // g                 # whole cycles per ARB block
    total_ticks = q * scale // g        # exact tick count for `num_cycles` periods

    min_ticks = max(1, math.ceil(NGU411_MIN_DWELL_S * scale))
    hard_max = min(NGU411_MAX_POINTS, total_ticks // min_ticks)
    pool = [d for d in _divisors(total_ticks) if 2 <= d <= hard_max]
    if not pool:
        raise ValueError(
            f"{float(f):g} Hz needs {num_cycles} whole period(s) = {total_ticks} "
            f"ticks of {1.0 / scale:g} s; no point count fits both the "
            f"{NGU411_MAX_POINTS}-point maximum and the {NGU411_MIN_DWELL_S * 1e3:g} ms "
            f"minimum dwell. Try a coarser frequency grid or a nearby frequency."
        )

    # Aim for 40 points per cycle.
    at_least = [d for d in pool if d >= 40 * num_cycles]
    num_points = min(at_least) if at_least else max(pool)

    if num_points < 4 * num_cycles:
        raise ValueError(
            f"{float(f):g} Hz needs {num_cycles} whole periods to avoid duration "
            f"rounding, but only {num_points} points fit ({num_points / num_cycles:.2g} "
            f"per cycle) -- too few to reproduce the sine. Pick a frequency on a "
            f"coarser grid (e.g. 0.5 Hz steps)."
        )

    return num_cycles, num_points, (total_ticks // num_points) / scale


def generate_ngu411_arb(f: Fraction, amplitude_v: float = 0.01, output_path: str = None) -> str:
    """Write an NGU411 ARB sine wave CSV and return its path."""
    num_cycles, n, duration_per_point = plan_arb(f)
    freq_val = float(f)
    dur_str = f"{duration_per_point:.{NGU411_DURATION_DECIMALS}f}"

    lines = [
        "#Device,NGU411",
        "#Format,ARB",
        "#Rep,0",
        "#EndBehavior,Output Off",
        "#PriorityMode,Voltage",
        "Voltage,Pos_Current,Neg_Current,Duration,Interp",
    ]
    # `num_cycles` whole cycles spread over `n` equally spaced samples.
    for i in range(n):
        voltage = amplitude_v * math.sin(2.0 * math.pi * num_cycles * i / n)
        lines.append(f"{voltage:.5f},0.0100000,-0.0100000,{dur_str},true")

    if output_path is None:
        output_path = f"{int(round(amplitude_v * 1000))}mV_{freq_val:g}Hz.csv"

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Generated: {output_path}")
    print(f"  Frequency   : {freq_val:g} Hz")
    print(f"  Cycles/block: {num_cycles}")
    print(f"  Points      : {n} total  ({n / num_cycles:.4g} per cycle)")
    print(f"  Dwell/point : {duration_per_point * 1e3:g} ms  ({dur_str} s)")
    print(f"  Block length: {n * duration_per_point * 1e3:.6f} ms  ({num_cycles} periods, loops via Rep)")
    if n / num_cycles < 12:
        print(f"  NOTE: only {n / num_cycles:.3g} points/cycle -- limited by the 0.1 ms "
              f"resolution at {freq_val:g} Hz; a nearby frequency may sample better")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate NGU411 ARB sine wave CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "frequency",
        type=Fraction,
        help="Sine wave frequency in Hz, parsed exactly (e.g. 10, 10.5, 21/2)",
    )
    parser.add_argument(
        "-a", "--amplitude",
        type=float,
        default=0.01,
        help="Peak amplitude in Volts (e.g. 0.01 for 10 mV)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (auto-generated if omitted)",
    )
    args = parser.parse_args()

    try:
        generate_ngu411_arb(args.frequency, args.amplitude, args.output)
    except (ValueError, ZeroDivisionError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
