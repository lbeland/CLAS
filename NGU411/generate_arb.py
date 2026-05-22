#!/usr/bin/env python3
"""
NGU411 Rohde & Schwarz Arbitrary Waveform File Generator
Generates sine wave .csv files compatible with the NGU411 ARB input format.
"""

import math
import argparse
import os


def generate_ngu411_arb(
    frequency_hz: float,
    amplitude_v: float = 0.01,
    pos_current_a: float = 0.01,
    neg_current_a: float = -0.01,
    num_points: int = 20,
    rep: int = 0,
    end_behavior: str = "Output Off",
    priority_mode: str = "Voltage",
    output_path: str = None,
) -> str:
    """
    Generate an NGU411 ARB waveform CSV file with a sine wave.

    Args:
        frequency_hz:    Desired sine wave frequency in Hz.
        amplitude_v:     Peak amplitude in Volts (default: 0.01 V = 10 mV).
        pos_current_a:   Positive current limit in Amps (default: 0.01 A).
        neg_current_a:   Negative current limit in Amps (default: -0.01 A).
        num_points:      Number of sample points per cycle (default: 20).
        rep:             Number of repetitions; 0 = infinite (default: 0).
        end_behavior:    Behavior when sequence ends (default: "Output Off").
        priority_mode:   Priority mode, "Voltage" or "Current" (default: "Voltage").
        output_path:     Output file path. Auto-generated if not provided.

    Returns:
        Path to the generated file.
    """
    # Duration per sample point so that num_points spans exactly one period
    period_s = 1.0 / frequency_hz
    duration_per_point = period_s / num_points

    # Build header
    lines = [
        f"#Device,NGU411",
        f"#Format,ARB",
        f"#Rep,{rep}",
        f"#EndBehavior,{end_behavior}",
        f"#PriorityMode,{priority_mode}",
        "Voltage,Pos_Current,Neg_Current,Duration,Interp",
    ]

    # Generate sine samples (one full cycle)
    for i in range(num_points):
        angle = 2.0 * math.pi * i / num_points
        voltage = amplitude_v * math.sin(angle)
        lines.append(
            f"{voltage:.5f},{pos_current_a:.7f},{neg_current_a:.7f},{duration_per_point:.4f},true"
        )

    content = "\n".join(lines) + "\n"

    # Auto-generate filename if not provided
    if output_path is None:
        amp_mv = int(round(amplitude_v * 1000))
        freq_label = (
            f"{int(frequency_hz)}Hz"
            if frequency_hz == int(frequency_hz)
            else f"{frequency_hz}Hz"
        )
        output_path = f"{amp_mv}mV_{freq_label}.csv"

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Generated: {output_path}")
    print(f"  Frequency  : {frequency_hz} Hz")
    print(f"  Period     : {period_s*1000:.4f} ms")
    print(f"  Amplitude  : {amplitude_v*1000:.3f} mV peak")
    print(f"  Points     : {num_points} per cycle")
    print(f"  Duration/pt: {duration_per_point*1000:.4f} ms")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate NGU411 ARB sine wave CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "frequency",
        type=float,
        help="Sine wave frequency in Hz (e.g. 10, 50, 1000)",
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
        help="Number of sample points per cycle",
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
    )


if __name__ == "__main__":
    main()