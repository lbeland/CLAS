import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(15, 8))

# Create histogram axis on the right
divider = make_axes_locatable(ax)
ax_hist = divider.append_axes("right", size="20%", pad=0.1)

for freq in ["0.5kHz", "1kHz", "5kHz", "10kHz"]:
    recv_times = pd.read_csv(f"receive_times_{freq}.csv", header=None).values.flatten() * 1e-6
    freq_value = len(recv_times)/5
    
    x = np.linspace(0, 5000, len(recv_times))

    ax_hist.hist(recv_times, bins=25, histtype='step', orientation='horizontal', label=freq)
    ax.plot(x, recv_times, label=freq)
    print(f"{freq} - Mean: {np.mean(recv_times):.3f} ms, Std: {np.std(recv_times):.3f} ms, Min: {np.min(recv_times):.3f} ms, Max: {np.max(recv_times):.3f} ms")


# recv_times = recv_times - 1e3/freq_value

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Receive Period (ms)")

ax.minorticks_on()
ax_hist.minorticks_on()
ax_hist.grid(which="both",axis="y")
ax.grid(which="both",axis="both")
ax.legend()
plt.savefig("TurboLink.png", dpi=300)
plt.show()
