#!/usr/bin/env python3
"""
NGU411 Rohde & Schwarz Arbitrary Waveform File Generator
Generates sine wave .csv files compatible with the NGU411 ARB input format.
"""

import math
import argparse
import os
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


def plan_arb(
    frequency_hz,
    target_points_per_cycle: int = 40,
    duration_decimals: int = NGU411_DURATION_DECIMALS,
    max_points: int = NGU411_MAX_POINTS,
    min_dwell_s: float = NGU411_MIN_DWELL_S,
):
    """
    Plan an ARB block that reproduces `frequency_hz` with NO rounding in the
    step duration.

    A single period of an arbitrary frequency almost never splits into equal
    steps that are exact at a finite decimal resolution (1/f is usually a
    repeating decimal). But an integer number of periods does: write
    ``f = p / q`` in lowest terms and let ``scale = 10**duration_decimals``
    (ticks per second). After ``M = p / gcd(p, scale)`` whole cycles the
    elapsed time is exactly ``total_ticks = q * scale / gcd(p, scale)`` ticks,
    which divides into equal, exact steps. The instrument's Rep loop repeats
    that block, so the average frequency is exactly `f` forever.

    `num_points` is chosen as the smallest divisor of `total_ticks` that gives
    at least `target_points_per_cycle` samples per cycle, subject to
    `num_points <= max_points` and dwell `>= min_dwell_s`; if none qualifies,
    the largest admissible divisor is used.

    Returns a dict with num_cycles, num_points, ticks_per_point, scale,
    total_ticks, frequency (Fraction).
    """
    f = frequency_hz if isinstance(frequency_hz, Fraction) else Fraction(str(frequency_hz))
    if f <= 0:
        raise ValueError("frequency must be positive")

    scale = 10 ** duration_decimals
    p, q = f.numerator, f.denominator
    g = math.gcd(p, scale)
    num_cycles = p // g                 # whole cycles per ARB block
    total_ticks = q * scale // g        # exact tick count for `num_cycles` periods

    min_ticks = max(1, math.ceil(min_dwell_s * scale))
    hard_max = min(max_points, total_ticks // min_ticks)
    pool = [d for d in _divisors(total_ticks) if 2 <= d <= hard_max]
    if not pool:
        hint = "a coarser frequency grid, a nearby frequency, or a larger --max-points"
        if duration_decimals < NGU411_DURATION_DECIMALS:
            hint = f"--duration-decimals {duration_decimals + 1}, " + hint
        raise ValueError(
            f"{float(f):g} Hz needs {num_cycles} whole period(s) = {total_ticks} "
            f"ticks of {1.0 / scale:g} s; no point count fits both the "
            f"{max_points}-point maximum and the {min_dwell_s * 1e3:g} ms minimum "
            f"dwell. Try {hint}."
        )

    target = max(2, round(target_points_per_cycle * num_cycles))
    at_least = [d for d in pool if d >= target]
    num_points = min(at_least) if at_least else max(pool)
    ticks_per_point = total_ticks // num_points

    if num_points < 4 * num_cycles:
        raise ValueError(
            f"{float(f):g} Hz needs {num_cycles} whole periods to avoid duration "
            f"rounding, but only {num_points} points fit ({num_points / num_cycles:.2g} "
            f"per cycle) -- too few to reproduce the sine. Pick a frequency on a "
            f"coarser grid (e.g. 0.5 Hz steps) or raise --max-points."
        )

    return {
        "num_cycles": num_cycles,
        "num_points": num_points,
        "ticks_per_point": ticks_per_point,
        "scale": scale,
        "total_ticks": total_ticks,
        "frequency": f,
    }


def generate_ngu411_arb(
    frequency_hz,
    amplitude_v: float = 0.01,
    pos_current_a: float = 0.01,
    neg_current_a: float = -0.01,
    num_points: int = 40,
    rep: int = 0,
    end_behavior: str = "Output Off",
    priority_mode: str = "Voltage",
    output_path: str = None,
    auto_points: bool = True,
    duration_decimals: int = NGU411_DURATION_DECIMALS,
    max_points: int = NGU411_MAX_POINTS,
) -> str:
    """
    Generate an NGU411 ARB waveform CSV file with a sine wave.

    Args:
        frequency_hz:    Desired sine wave frequency in Hz (number, Fraction, or
                         string like "10.5" / "21/2").
        amplitude_v:     Peak amplitude in Volts (default: 0.01 V = 10 mV).
        pos_current_a:   Positive current limit in Amps (default: 0.01 A).
        neg_current_a:   Negative current limit in Amps (default: -0.01 A).
        num_points:      With auto_points, the TARGET samples per cycle; the
                         block gets the nearest exact-duration count, possibly
                         spanning several cycles. Without auto_points, the exact
                         number of samples for one cycle (default: 40).
        rep:             Number of repetitions; 0 = infinite (default: 0).
        end_behavior:    Behavior when sequence ends (default: "Output Off").
        priority_mode:   Priority mode, "Voltage" or "Current" (default: "Voltage").
        output_path:     Output file path. Auto-generated if not provided.
        auto_points:     Pick a cycle count / point count so the step duration
                         has no rounding at duration_decimals places (default:
                         True).
        duration_decimals: Decimal places written in the Duration column;
                         resolution is 10**-duration_decimals s (default: 4,
                         i.e. 0.1 ms, the NGU411 finest step). Each dwell is
                         still kept >= 1 ms (the NGU411 minimum).
        max_points:      Cap on total samples in the file (default: 4096).

    Returns:
        Path to the generated file.
    """
    f = frequency_hz if isinstance(frequency_hz, Fraction) else Fraction(str(frequency_hz))
    freq_val = float(f)
    period_s = 1.0 / freq_val

    if auto_points:
        plan = plan_arb(
            f,
            target_points_per_cycle=num_points,
            duration_decimals=duration_decimals,
            max_points=max_points,
        )
        num_cycles = plan["num_cycles"]
        n = plan["num_points"]
        duration_per_point = plan["ticks_per_point"] / plan["scale"]
    else:
        num_cycles = 1
        n = num_points
        duration_per_point = period_s / n

    dur_str = f"{duration_per_point:.{duration_decimals}f}"

    # Build header
    lines = [
        f"#Device,NGU411",
        f"#Format,ARB",
        f"#Rep,{rep}",
        f"#EndBehavior,{end_behavior}",
        f"#PriorityMode,{priority_mode}",
        "Voltage,Pos_Current,Neg_Current,Duration,Interp",
    ]

    # `num_cycles` whole cycles spread over `n` equally spaced samples.
    for i in range(n):
        angle = 2.0 * math.pi * num_cycles * i / n
        voltage = amplitude_v * math.sin(angle)
        lines.append(
            f"{voltage:.5f},{pos_current_a:.7f},{neg_current_a:.7f},{dur_str},true"
        )

    content = "\n".join(lines) + "\n"

    # Auto-generate filename if not provided
    if output_path is None:
        amp_mv = int(round(amplitude_v * 1000))
        freq_label = (
            f"{int(freq_val)}Hz" if freq_val == int(freq_val) else f"{freq_val:g}Hz"
        )
        output_path = f"{amp_mv}mV_{freq_label}.csv"

    with open(output_path, "w") as fh:
        fh.write(content)

    written = float(dur_str)
    block_s = n * written
    eff_freq = num_cycles / block_s if block_s else float("nan")
    freq_err = eff_freq - freq_val

    print(f"Generated: {output_path}")
    print(f"  Frequency   : {freq_val:g} Hz")
    print(f"  Cycles/block: {num_cycles}")
    print(f"  Points      : {n} total  ({n / num_cycles:.4g} per cycle)")
    print(f"  Dwell/point : {written * 1e3:g} ms  ({dur_str} s)")
    print(f"  Block length: {block_s * 1e3:.6f} ms  ({num_cycles} periods, loops via Rep)")
    if abs(freq_err) < 1e-9:
        print(f"  Effective f : {eff_freq:.10g} Hz  (exact)")
    else:
        print(f"  Effective f : {eff_freq:.10g} Hz  (off by {freq_err:+.3e} Hz -- ROUNDED)")

    if written < NGU411_MIN_DWELL_S - 1e-12:
        print(f"  WARNING: dwell {written * 1e3:g} ms < NGU411 minimum {NGU411_MIN_DWELL_S * 1e3:g} ms")
    if n > NGU411_MAX_POINTS:
        print(f"  WARNING: {n} points > NGU411 maximum {NGU411_MAX_POINTS}")
    if n / num_cycles < 12:
        print(f"  NOTE: only {n / num_cycles:.3g} points/cycle -- limited by the 0.1 ms "
              f"resolution and 1 ms min dwell at {freq_val:g} Hz; a nearby frequency "
              f"may sample better")
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
        "-p", "--pos-current",
        type=float,
        default=0.01,
        dest="pos_current",
        help="Positive current limit in Amps",
    )
    parser.add_argument(
        "-n", "--neg-current",
        type=float,
        default=-0.01,
        dest="neg_current",
        help="Negative current limit in Amps",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=40,
        help="Target sample points per cycle; the block gets the nearest "
             "exact-duration count (may span several cycles) unless "
             "--no-auto-points",
    )
    parser.add_argument(
        "--no-auto-points",
        action="store_true",
        help="Emit exactly one cycle with --points samples, even if the step "
             "duration then has to be rounded",
    )
    parser.add_argument(
        "--duration-decimals",
        type=int,
        default=NGU411_DURATION_DECIMALS,
        help="Decimal places for the Duration column (resolution 10**-d s); "
             "NGU411 finest is 4 (0.1 ms)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=NGU411_MAX_POINTS,
        help="Cap on total samples in the file",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=0,
        help="Repetitions (0 = infinite)",
    )
    parser.add_argument(
        "--end-behavior",
        type=str,
        default="Output Off",
        help='End behavior string (e.g. "Output Off", "Hold Last Value")',
    )
    parser.add_argument(
        "--priority-mode",
        type=str,
        default="Voltage",
        choices=["Voltage", "Current"],
        help="Priority mode",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (auto-generated if omitted)",
    )

    args = parser.parse_args()

    try:
        generate_ngu411_arb(
            frequency_hz=args.frequency,
            amplitude_v=args.amplitude,
            pos_current_a=args.pos_current,
            neg_current_a=args.neg_current,
            num_points=args.points,
            rep=args.rep,
            end_behavior=args.end_behavior,
            priority_mode=args.priority_mode,
            output_path=args.output,
            auto_points=not args.no_auto_points,
            duration_decimals=args.duration_decimals,
            max_points=args.max_points,
        )
    except (ValueError, ZeroDivisionError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()