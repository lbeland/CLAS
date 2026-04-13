import math
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
# import pyqtgraph as pg
# from PyQt5.QtWidgets import QApplication


results_folder = "rt_c_results"
send_files = [f for f in os.listdir(results_folder) if "send_times" in f]
recv_files = [f for f in os.listdir(results_folder) if "recv_times" in f]
latency_files = [f for f in os.listdir(results_folder) if "process_times" in f]

fig, axes = plt.subplots(3, 1,sharex=True)
fig.suptitle("Falcon - Performance Over Time", fontsize=14, fontweight='bold')
axes = np.atleast_1d(axes).ravel()

for idx, x in enumerate(zip([latency_files, send_files, recv_files], ["Latency", "Send Period", "Receive Period"])):
    files, label = x
    ax = axes[idx]
    ax.set_ylabel(f"{label} (μs)")
    ax.set_xlabel("Message Index")
    ax.set_yscale('log')
    # ax.set_ylim(0.1, 5000)

    # create histogram axis to the right
    divider = make_axes_locatable(ax)
    ax_hist = divider.append_axes("right", size="20%", pad=0.1, sharey=ax)

    for file in sorted(files):
        
        n_channels, msg_size, node, _ = file.split('_')
        if int(n_channels) == 20 and int(msg_size) >= 1024:
            pass
        else:
            print(f"Processing file: {file} | Channels: {n_channels} | Msg Size: {msg_size} ")

            data = np.loadtxt(os.path.join(results_folder, file), skiprows=0, delimiter=',')
            ax.plot(data[:]*1e-3, linewidth=0.5,label=f"Ch {n_channels} | Msg {msg_size}")

            # Use logarithmic bins for better distribution when data spans multiple orders of magnitude
            data_us = data[:]*1e-3  # convert to microseconds
            bins = np.logspace(np.log10(max(data_us.min(), 0.01)), np.log10(data_us.max()), 20)
            n,bins,_ = ax_hist.hist(data_us, bins=bins, histtype='step',orientation='horizontal')
            ax_hist.set_xlabel("Frequency")
            ax_hist.set_yscale('log')
            # ax_hist.set_xscale('log')
            # ax_hist.grid(True, which="both", alpha=0.3)


axes[0].legend(loc="upper right")
plt.savefig('performance_over_time.png', dpi=300, bbox_inches='tight')
plt.show()



# app = QApplication([])

# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

# win = pg.GraphicsLayoutWidget(show=True, title="Falcon Benchmark")
# win.resize(1000, 800)

# plots = []

# titles = ["Latency", "Send Period", "Receive Period"]
# file_groups = [latency_files, send_files, recv_files]

# for idx, (files, label) in enumerate(zip(file_groups, titles)):

#     plot = win.addPlot(row=idx, col=0, title=label)
#     plot.setLabel("left", "Time (μs)")
#     plot.setLabel("bottom", "Message Index")
#     plot.addLegend()

#     plots.append(plot)

#     for idx, file in enumerate(sorted(files)):

#         n_channels, msg_size, node, _ = file.split('_')
#         print(f"Processing file: {file} | Channels: {n_channels} | Msg Size: {msg_size}")

#         data = np.loadtxt(os.path.join(results_folder, file),
#                           skiprows=1, delimiter=',')

#         y = data[:] * 1e6
#         x = np.arange(len(y))
        
#         curve = plot.plot(
#             x,
#             y,
#             pen=pg.mkPen(list(colors.TABLEAU_COLORS.values())[idx % len(colors.TABLEAU_COLORS)], width=2),
#             name=f"Ch {n_channels} | Msg {msg_size}",
#             clickable=True
#         )

#         def on_click(item, file=file, ch=n_channels, msg=msg_size):
#             print(f"Clicked curve → file={file}, channels={ch}, msg_size={msg}")

#         curve.sigClicked.connect(on_click)

# app.exec()