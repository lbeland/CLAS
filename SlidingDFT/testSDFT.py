import numpy as np
from sdft import SDFT
import matplotlib.pyplot as plt
import time


signal_length = 510
window_length = 500
hop = 1
n_iterations = signal_length - window_length  # 2000

fs = 1000.0
t = np.arange(signal_length) / fs

# Example: 10 Hz sine wave
x = np.sin(2 * np.pi * 10 * t)


m = window_length//2  # DFT size (and SDFT internal state size)
sdft = SDFT(m, window='boxcar', latency=1)
L = window_length

first_window = x[:window_length]

# Initial spectrum from the first full window
X_sdft = sdft.sdft(first_window)
y_sdft = sdft.isdft(X_sdft)

# Store spectra if you want
errors = []
y_sdfts = []
y_fulls = []

print(f"Initial spectrum shape: {X_sdft.shape}")

fig_freq, ax_freq = plt.subplots(num='Frequency Domain')
fig_time, ax_time = plt.subplots(num='Time Domain')
ax_time.plot(x, '--',label='Input Signal')


# --- Sliding DFT ---
start_time  = time.time()
for start in range(1, signal_length - L):

    new_sample = x[start + L - 1]
    X_sdft = sdft.sdft([new_sample])[0]
    y_sdft = sdft.isdft(X_sdft)[0]

    y_sdfts.append(y_sdft)
    ax_freq.plot(X_sdft, 'o', alpha=0.5)
    ax_time.plot(window_length + start - 1, y_sdft, 'o', alpha=0.5)
end_time  = time.time()
print(f"SDFT processing time for {n_iterations} updates: {end_time  - start_time :.4f} seconds")

# --- Full FFT for comparison ---
start_time  = time.time()
for start in range(1, signal_length - L):
    
    window = x[start:start + L]
    X_full = np.fft.rfft(window)
    y_full = np.fft.irfft(X_full)[-1]

    y_fulls.append(y_full)

    ax_freq.plot(X_full/L, 'x', alpha=0.5)
    ax_time.plot(window_length + start - 1, y_full, 'x', alpha=0.5)
end_time  = time.time()
print(f"Full FFT processing time for {n_iterations} updates: {end_time  - start_time :.4f} seconds")


errors = np.array(y_sdfts) - np.array(y_fulls)
# plt.figure()
# plt.plot(errors)

print(f"Sliding updates performed: {n_iterations}")  # 2000
print("Mean absolute difference between SDFT and full DFT outputs:", np.mean(np.abs(errors)))
print("Max absolute difference between SDFT and full DFT outputs:", np.max(np.abs(errors)))
print("Standard deviation of difference between SDFT and full DFT outputs:", np.std(errors))

plt.show()