import os
import sys

import numpy as np
from sdft import SDFT
from sdft import STFT
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from IAF_Estimation.cecHT.phase import ECHT
import time

np.random.seed(0)  # For reproducibility

signal_length = 30010
window_length = 10000 # two full cycle
n_iterations = signal_length - window_length

fs = 10000.0
t = np.arange(signal_length) / fs

# Example: 10 Hz sine wave
f0 = 9.3
x = np.cos(2 * np.pi * f0 * t) # + 0.2 * np.random.rand(signal_length)  # one cycle every 1000 samples
true_phase = np.angle(np.exp(1j * 2 * np.pi * f0 * t))

k0 = int(np.round(f0 * window_length / fs))
bin_freq = k0 * fs / window_length
mismatch = f0 - bin_freq

m = window_length//2 # + 1  # DFT size (and SDFT internal state size)
sdft = SDFT(m, window='boxcar', latency=1)

first_window = x[:window_length]

# Initial spectrum from the first full window
X_sdft = sdft.sdft(first_window)
y_sdft = sdft.isdft(X_sdft)

# Store spectra if you want
errors = []
phases = []


# --- Sliding DFT ---
start_time  = time.time()
for start in range(1, signal_length - window_length+1):
    new_sample = x[start + window_length - 1]
    X_sdft = sdft.sdft([new_sample])[0] # length window_length//2
    # x_sdfts.append(X_sdft)
    phases.append(np.angle(X_sdft[k0]))
    
end_time  = time.time()
print(f"SDFT processing time per iteration: {(end_time  - start_time) / n_iterations * 1000:.4f} ms")

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(211)
ax1.plot(t,x, label='Signal')
ax1.plot(t[window_length:], phases, label='SDFT Phase at 10 Hz', color='orange')
ax1.plot(t, true_phase, label='True Phase', color='red', linestyle='dashed')
ax1.legend()
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(t[window_length:], np.angle(np.exp(1j * (phases - true_phase[window_length:]))), label='Phase error', color='green')
ax2.legend()


plt.savefig("sdft_vs_fullfft.png", dpi=300)
plt.show()