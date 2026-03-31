import numpy as np
from sdft import SDFT
import matplotlib.pyplot as plt
import time


signal_length = 2100
window_length = 1000 # one full cycle
n_iterations = signal_length - window_length  # 2000

fs = 10000.0
t = np.arange(signal_length) / fs

# Example: 10 Hz sine wave
x = np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.rand(signal_length)  # one cycle every 1000 samples


m = window_length//2  # DFT size (and SDFT internal state size)
sdft = SDFT(m, window='boxcar', latency=1)

first_window = x[:window_length]

# Initial spectrum from the first full window
X_sdft = sdft.sdft(first_window)
y_sdft = sdft.isdft(X_sdft)

# Store spectra if you want
errors = []
y_sdfts = []
y_fulls = []

# fig_freq, ax_freq = plt.subplots(num='Frequency Domain')
fig_time, ax_time = plt.subplots(num='Time Domain')
ax_time.plot(x, '--',label='Input Signal')


# --- Sliding DFT ---
start_time  = time.time()
for start in range(1, signal_length - window_length):

    new_sample = x[start + window_length - 1]
    X_sdft = sdft.sdft([new_sample])[0] # length window_length//2
    y_sdft = sdft.isdft(X_sdft)[0]

    y_sdfts.append(y_sdft)
    # ax_freq.plot(X_sdft, 'o', alpha=0.5)
    ax_time.plot(window_length + start - 1, y_sdft, 'o', color='blue', alpha=0.5)
end_time  = time.time()
print(f"SDFT processing time per iteration: {(end_time  - start_time) / n_iterations * 1000:.4f} ms")

# --- Full FFT for comparison ---
start_time  = time.time()
for start in range(1, signal_length - window_length):
    
    window = x[start:start + window_length]
    X_full = np.fft.rfft(window)    # length window_length//2 + 1
    y_full = np.fft.irfft(X_full)[-1]

    y_fulls.append(y_full)

    # ax_freq.plot(X_full/window_length, 'x', alpha=0.5)
    ax_time.plot(window_length + start - 1, y_full, 'x', color='red', alpha=0.5)
end_time  = time.time()
print(f"Full FFT processing time per iteration: {(end_time  - start_time) / n_iterations * 1000:.4f} ms")


errors = np.array(y_sdfts) - np.array(y_fulls)
# plt.figure()
# plt.plot(errors)

print(f"Iterations performed: {n_iterations}")  # 2000

if len(errors) > 0:
    print("Mean absolute difference between SDFT and full DFT outputs:", np.mean(np.abs(errors)))
    print("Max absolute difference between SDFT and full DFT outputs:", np.max(np.abs(errors)))
    print("Standard deviation of difference between SDFT and full DFT outputs:", np.std(errors))

plt.show()