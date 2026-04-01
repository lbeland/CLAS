import numpy as np
from sdft import SDFT
from sdft import STFT
import matplotlib.pyplot as plt
import time

np.random.seed(0)  # For reproducibility

signal_length = 8002
window_length = 1000 # one full cycle
n_iterations = signal_length - window_length  # 2000

fs = 10000.0
t = np.arange(signal_length) / fs

# Example: 10 Hz sine wave
x = np.sin(2 * np.pi * 10 * t) # + 0.2 * np.random.rand(signal_length)  # one cycle every 1000 samples


m = window_length//2 # + 1  # DFT size (and SDFT internal state size)
sdft = SDFT(m, window='boxcar', latency=1)

first_window = x[:window_length]

# Initial spectrum from the first full window
X_sdft = sdft.sdft(first_window)
y_sdft = sdft.isdft(X_sdft)

# Store spectra if you want
errors = []

y_sdfts = []
x_sdfts = []
y_fulls = []
x_fulls = []

# fig_freq, ax_freq = plt.subplots(num='Frequency Domain')
# fig_time, ax_time = plt.subplots(num='Time Domain')
# ax_time.plot(x, '--',label='Input Signal')


# --- Sliding DFT ---
start_time  = time.time()
# dfts = sdft.sdft(x)
for start in range(1, signal_length - window_length):
    new_sample = x[start + window_length - 1]
    X_sdft = sdft.sdft([new_sample])[0] # length window_length//2
    x_sdfts.append(X_sdft)
    
    y_sdft = sdft.isdft(X_sdft)[0]
    y_sdfts.append(y_sdft)
    # ax_freq.plot(np.abs(X_sdft), 'ro-', alpha=0.5)
    # ax_time.plot(window_length + start - 1, y_sdft, 'o', color='blue', alpha=0.5)
end_time  = time.time()
print(f"SDFT processing time per iteration: {(end_time  - start_time) / n_iterations * 1000:.4f} ms")

# --- Shorttime Fourier Transform ---
# stft = STFT(window_length, 1, window='boxcar')
# start_time  = time.time()
# dfts = stft.stft(x)
# end_time  = time.time()
# print(f"STFT processing time per iteration: {(end_time  - start_time) / n_iterations * 1000:.4f} ms")


# --- Full FFT for comparison ---
start_time  = time.time()
for start in range(1, signal_length - window_length):
    window = x[start:start + window_length]
    X_full = np.fft.rfft(window)    # length window_length//2 + 1
    x_fulls.append(X_full[0:-1]/window_length)

    y_full = np.fft.irfft(X_full)[-1]
    y_fulls.append(y_full)

    # ax_freq.plot(np.abs(X_full)/window_length, 'bx-', alpha=0.5)
    # ax_time.plot(window_length + start - 1, y_full, 'x', color='red', alpha=0.5)
end_time  = time.time()
print(f"Full FFT processing time per iteration: {(end_time  - start_time) / n_iterations * 1000:.4f} ms")

y_sdfts = np.array(y_sdfts)
x_sdfts = np.abs(np.array(x_sdfts))
y_fulls = np.array(y_fulls)
x_fulls = np.abs(np.array(x_fulls))

errors_time = y_sdfts - y_fulls

print(f"Shape of errors: {x_sdfts.shape}, {x_fulls.shape}")
errors_freq = np.mean(x_sdfts - x_fulls, axis=1)  # Average error across all bins per iteration
plt.figure()
plt.plot(errors_time)
plt.plot(errors_freq)

print(f"Iterations performed: {n_iterations}")  # 2000

if len(errors_time) > 0:
    print("----------- Time -----------")
    print("Mean absolute difference between SDFT and full DFT outputs:", np.mean(np.abs(errors_time)))
    print("Max absolute difference between SDFT and full DFT outputs:", np.max(np.abs(errors_time)))
    print("Standard deviation of difference between SDFT and full DFT outputs:", np.std(errors_time))

if len(errors_freq) > 0:
    print("----------- Freq ------------")
    print("Mean absolute difference between SDFT and full DFT outputs:", np.mean(np.abs(errors_freq)))
    print("Max absolute difference between SDFT and full DFT outputs:", np.max(np.abs(errors_freq)), np.argmax(np.abs(errors_freq)))
    print("Standard deviation of difference between SDFT and full DFT outputs:", np.std(errors_freq))

plt.show()