import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_FIGURES = _HERE / "figures"

fooof_results = pd.read_csv(_RESULTS / "results_ds004148_fooof_Pz" / "iaf_per_segment.csv")
myIAF_results = pd.read_csv(_RESULTS / "results_ds004148_myIAF_Pz" / "iaf_per_segment.csv")

merge_column = ["file", "segment_index", "time_s"]
merged = pd.merge(
    fooof_results,
    myIAF_results,
    left_on=merge_column,
    right_on=merge_column,
    suffixes=("_fooof", "_myIAF"),
)

print(f"Merged {len(merged)} segments with IAF estimates from both methods")

# Print rows where IAF estimates differ by more than 0.5 Hz
diff = np.abs(merged["paf_hz_fooof"] - merged["paf_hz_myIAF"])
large_diff_mask = diff > 2
if large_diff_mask.any():
    print(f"{large_diff_mask.sum()} segments with large IAF differences (>2 Hz):")
    print(merged.loc[large_diff_mask, merge_column + ["paf_hz_fooof", "paf_hz_myIAF"]])


diff = merged["paf_hz_fooof"] - merged["paf_hz_myIAF"]

mean_paf = (merged["paf_hz_fooof"] + merged["paf_hz_myIAF"]) / 2
mean_diff = diff.mean()
sd_diff = diff.std()
loa_upper = mean_diff + 1.96 * sd_diff
loa_lower = mean_diff - 1.96 * sd_diff

print(f"Mean difference: {mean_diff:.3f} Hz")
print(f"Std difference: {sd_diff:.3f} Hz")

# 2d hist 
plt.figure()
plt.hist2d(merged["paf_hz_fooof"], merged["paf_hz_myIAF"], bins=100, cmap="terrain")
plt.colorbar(label="Count")
plt.xlabel("FOOOF IAF (Hz)")
plt.ylabel("My IAF (Hz)")

plt.figure()
plt.hist(diff, bins=60, edgecolor="black")
plt.xlabel("FOOOF IAF - My IAF (Hz)")
plt.ylabel("Number of segments")
plt.title("Distribution of IAF differences")
plt.grid()
plt.savefig(_FIGURES / "iaf_difference_histogram.png")

plt.figure()
plt.plot(merged["paf_hz_fooof"], merged["paf_hz_myIAF"], "o", markersize=3, alpha=0.5)
plt.xlabel("FOOOF IAF (Hz)")
plt.ylabel("My IAF (Hz)")

plt.xlim(6, 15)
plt.ylim(6, 15)
plt.title("Comparison of IAF estimates")
plt.grid()
plt.savefig(_FIGURES / "iaf_comparison.png")
plt.show()
