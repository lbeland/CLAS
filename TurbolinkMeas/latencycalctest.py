"""
Verification of the SourceClient calibration logic.

Replicates the C++ formula:

    offset_i        = timestamp_i - (sample_counter_i - first_counter) / fs
    start_time_us_  = min(offset_i)
    hardware_time_us(n) = start_time_us_ + n / fs

and checks:
  1. start_time_us_ converges to (true_start_time + fixed_delay) as the
     calibration window grows (since min(jitter) -> 0).
  2. hardware_time_us(n) is an exactly linear function of n with slope 1/fs.
  3. hardware_time_us(n) <= timestamp_n always (lower-bound property).
  4. The calibration still works correctly even when fixed_delay is many
     multiples of the inter-packet interval (1/fs) -- a constant pipeline
     fill delay does not break the linear mapping.

Two arrival models are provided:
  - simulate_independent: each packet's jitter is i.i.d, timestamps may be
    "out of order" relative to an idealized FIFO (fine for short jitter
    relative to 1/fs).
  - simulate_fifo: enforces non-decreasing delivery order, i.e. a packet
    cannot be delivered before the previous one. This matters when jitter
    is comparable to or larger than the inter-packet interval (1/fs),
    which is exactly the case for the "avg 1ms / max 1.5ms" spec at 10 kHz
    (interval = 100 us).
"""

import numpy as np


def calibrate(sample_counter, timestamp_us, fs):
    """Replicates the C++ calibration: returns start_time_us and hardware_time_us(n)."""
    first_counter = sample_counter[0]
    rel_samples = sample_counter - first_counter
    offset_us = timestamp_us - rel_samples / fs * 1e6
    start_time_us = offset_us.min()
    hardware_time_us = start_time_us + rel_samples / fs * 1e6
    return start_time_us, hardware_time_us, offset_us


def simulate_independent(n_packets, fs, fixed_delay_us, jitter_max_us,
                          true_start_time_us=1e9, seed=0, jitter_dist="uniform"):
    rng = np.random.default_rng(seed)
    sample_counter = np.arange(n_packets, dtype=np.int64)
    ideal_time_us = sample_counter / fs * 1e6

    if jitter_dist == "uniform":
        jitter_us = rng.uniform(0.0, jitter_max_us, size=n_packets)
    elif jitter_dist == "exponential":
        jitter_us = rng.exponential(jitter_max_us, size=n_packets)
    else:
        raise ValueError("unknown jitter_dist")

    timestamp_us = true_start_time_us + ideal_time_us + fixed_delay_us + jitter_us
    return sample_counter, timestamp_us, jitter_us


def simulate_fifo(n_packets, fs, fixed_delay_us, jitter_max_us,
                   true_start_time_us=1e9, seed=0, jitter_dist="uniform"):
    """Like simulate_independent, but enforces non-decreasing delivery order
    (a packet cannot be received before the previous one)."""
    sample_counter, raw_timestamp_us, jitter_us = simulate_independent(
        n_packets, fs, fixed_delay_us, jitter_max_us, true_start_time_us, seed, jitter_dist
    )

    delivery_us = np.empty(n_packets)
    delivery_us[0] = raw_timestamp_us[0]
    for i in range(1, n_packets):
        delivery_us[i] = max(raw_timestamp_us[i], delivery_us[i - 1])

    return sample_counter, delivery_us, jitter_us


def verify(sample_counter, timestamp_us, fs, fixed_delay_us, true_start_time_us, label=""):
    start_time_us, hardware_time_us, offset_us = calibrate(sample_counter, timestamp_us, fs)

    true_floor_us = true_start_time_us + fixed_delay_us
    floor_error_us = start_time_us - true_floor_us  # should be >= 0, -> 0 as N grows

    # Linearity check: hardware_time_us should advance by exactly 1e6/fs per sample
    diffs = np.diff(hardware_time_us)
    expected_step = 1e6 / fs
    max_linearity_error_us = np.max(np.abs(diffs - expected_step))

    # Lower-bound check: hardware_time_us(n) must never exceed timestamp_n
    lower_bound_violations = np.sum(hardware_time_us > timestamp_us + 1e-9)

    # "True" per-packet latency this represents (hardware_time_us vs actual sample time)
    true_sample_time_us = true_start_time_us + sample_counter / fs * 1e6
    implied_latency_us = hardware_time_us - true_sample_time_us  # should == fixed_delay_us (constant)

    print(f"--- {label} ---")
    print(f"  N packets:                 {len(sample_counter)}")
    print(f"  fs:                        {fs} Hz  (interval = {1e6/fs:.2f} us)")
    print(f"  fixed_delay_us (truth):    {fixed_delay_us:.3f}")
    print(f"  start_time_us_ (estimate): {start_time_us - true_start_time_us:.3f}  (relative to true_start_time)")
    print(f"  floor estimation error:    {floor_error_us:.3f} us  (= min observed jitter, should -> 0)")
    print(f"  max linearity error:       {max_linearity_error_us:.6f} us (should be ~0)")
    print(f"  lower-bound violations:    {lower_bound_violations} (should be 0)")
    print(f"  implied constant latency:  mean={implied_latency_us.mean():.3f} us, "
          f"std={implied_latency_us.std():.6f} us (should equal fixed_delay_us with ~0 std)")
    print()

    return start_time_us, hardware_time_us


if __name__ == "__main__":
    TRUE_START = 1_000_000_000.0  # arbitrary large "steady_clock epoch" offset, in us

    # ------------------------------------------------------------------
    # 1) Basic sanity check: small fixed delay, jitter << interval
    # ------------------------------------------------------------------
    sc, ts, jit = simulate_independent(
        n_packets=2000, fs=10000.0, fixed_delay_us=64.0, jitter_max_us=20.0,
        true_start_time_us=TRUE_START, jitter_dist="uniform",
    )
    verify(sc, ts, fs=10000.0, fixed_delay_us=64.0, true_start_time_us=TRUE_START,
           label="Basic: fs=10kHz, fixed=64us, jitter<=20us")

    # ------------------------------------------------------------------
    # 2) Convergence with calibration window size N
    #    (floor estimation error = min(jitter) -> 0 as N grows)
    # ------------------------------------------------------------------
    print("--- Convergence vs calibration window size N ---")
    for n in [10, 100, 1000, 10000]:
        sc, ts, jit = simulate_independent(
            n_packets=n, fs=10000.0, fixed_delay_us=64.0, jitter_max_us=64.0,
            true_start_time_us=TRUE_START, seed=42, jitter_dist="uniform",
        )
        start_time_us, _, _ = calibrate(sc, ts, fs=10000.0)
        err = start_time_us - (TRUE_START + 64.0)
        print(f"  N={n:6d}  floor error = {err:7.3f} us  (theoretical min over N draws of U(0,64) "
              f"~ {64.0 / (n + 1):.3f} us)")
    print()

    # ------------------------------------------------------------------
    # 3) Manual spec scenario: avg=1ms, max=1.5ms  =>  floor=0.5ms, jitter~U(0,1ms)
    #    Test at both 10 kHz (interval=100us, delay = 10x interval) and
    #    100 kHz (interval=10us, delay = 100x interval), using the FIFO model
    #    since jitter is comparable to/larger than the interval.
    # ------------------------------------------------------------------
    FIXED_DELAY = 500.0   # us  (floor)
    JITTER_MAX = 1000.0   # us  (uniform jitter on top of floor)
    # -> total delay: min=500us, mean=1000us, max=1500us  (matches manual)

    for fs in [10000.0, 100000.0]:
        sc, ts, jit = simulate_fifo(
            n_packets=50000, fs=fs, fixed_delay_us=FIXED_DELAY, jitter_max_us=JITTER_MAX,
            true_start_time_us=TRUE_START, seed=7, jitter_dist="uniform",
        )
        total_delay_us = jit + FIXED_DELAY
        print(f"  [fs={fs:.0f} Hz] total physical delay: "
              f"min={total_delay_us.min():.1f}, mean={total_delay_us.mean():.1f}, "
              f"max={total_delay_us.max():.1f} us  "
              f"(delay/interval ratio: {FIXED_DELAY/(1e6/fs):.1f}x to "
              f"{(FIXED_DELAY+JITTER_MAX)/(1e6/fs):.1f}x)")
        verify(sc, ts, fs=fs, fixed_delay_us=FIXED_DELAY, true_start_time_us=TRUE_START,
               label=f"Manual spec (FIFO): fs={fs:.0f}Hz, floor=500us, jitter<=1000us")

    # ------------------------------------------------------------------
    # 4) Exponential jitter (heavier tail) for comparison
    # ------------------------------------------------------------------
    sc, ts, jit = simulate_fifo(
        n_packets=10000, fs=10000.0, fixed_delay_us=500.0, jitter_max_us=500.0,
        true_start_time_us=TRUE_START, seed=99, jitter_dist="exponential",
    )
    verify(sc, ts, fs=10000.0, fixed_delay_us=500.0, true_start_time_us=TRUE_START,
           label="Exponential jitter: fs=10kHz, floor=500us, jitter~Exp(mean=500us)")