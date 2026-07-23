import numpy as np
import matplotlib.pyplot as plt

# Colors from the repo's validated dataviz palette (references/palette.md):
# one hue (blue), light step for the raw/noisy series, dark step for the
# smoothed trend -- raw and trend are the same metric, not different series.
RAW_COLOR   = "#86b6ef"   # sequential step 250
TREND_COLOR = "#184f95"   # sequential step 600
GRID_COLOR  = "#e1e0d9"
AXIS_COLOR  = "#c3c2b7"
TEXT_COLOR  = "#0b0b0b"
MUTED_COLOR = "#898781"

errors = np.fromfile("sdft_error_growth.bin", dtype=np.float64)
n = errors.size
iterations = np.arange(n)

# Block-average into a manageable number of points for a readable trend line.
n_blocks = 2000
block_size = n // n_blocks
trimmed = errors[: n_blocks * block_size].reshape(n_blocks, block_size)
block_mean = trimmed.mean(axis=1)
block_x = iterations[: n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1)

# Linear fit on the raw (unblocked) series to quantify drift: slope in
# error-units per iteration. If the recursive sdft state were accumulating
# numerical error over time, this slope would be clearly positive and the
# fit would visibly rise across the plot.
slope, intercept = np.polyfit(iterations, errors, 1)
fit_line = slope * block_x + intercept

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

ax.plot(iterations, errors, color=RAW_COLOR, linewidth=0.5, alpha=0.6,
        label="per-iteration error")
ax.plot(block_x, block_mean, color=TREND_COLOR, linewidth=2,
        label=f"block mean (n={block_size})")
ax.plot(block_x, fit_line, color=TEXT_COLOR, linewidth=1.2, linestyle="--",
        label=f"linear fit (slope = {slope:.2e} / iteration)")

ax.set_xlabel("iteration (samples processed)", color=TEXT_COLOR)
ax.set_ylabel("mean per-bin |mSDFT − FFT| magnitude error", color=TEXT_COLOR)
ax.set_title("mSDFT frequency-domain error vs. time (2,000,000 iterations)",
             color=TEXT_COLOR)

ax.set_yscale("log")
ax.grid(True, color=GRID_COLOR, linewidth=0.7)
for spine in ax.spines.values():
    spine.set_color(AXIS_COLOR)
ax.tick_params(colors=MUTED_COLOR)

legend = ax.legend(frameon=False, labelcolor=TEXT_COLOR)

fig.tight_layout()
fig.savefig("sdft_error_growth.png", dpi=150)

print(f"slope: {slope:.6e} error/iteration")
print(f"projected error after 1e9 iterations from fit: {slope * 1e9 + intercept:.3e}")
print(f"mean error overall: {errors.mean():.3e}")
