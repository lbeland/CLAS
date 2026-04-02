import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meegkit.phase import ECHT
from scipy.signal import hilbert
from read_output import get_signal_data

# Load data
samples_orig = get_signal_data('rt_c_results/Serializer1.0_Producer.out.0.bin')[:,0]
samples_filtered = get_signal_data('rt_c_results/Serializer2.0_BandpassFilter.out.0.bin')[:,0]
samples_phase = get_signal_data('rt_c_results/Serializer3.0_PhaseEstimator.out.0.bin')[:,0]
samples_real = get_signal_data('rt_c_results/Serializer3.1_PhaseEstimator.out.1.bin')[:,0]

fs = 1000
f0 = 8.5
filt_BW = f0 / 2
l_freq = f0 - filt_BW / 2
h_freq = f0 + filt_BW / 2

cecht = ECHT(l_freq, h_freq, fs, filt_order=1, calibrate=True, f0=f0)
cecht_Xf = cecht.fit_transform(samples_orig)
cecht_phase = np.angle(cecht_Xf)

echt = ECHT(l_freq, h_freq, fs, filt_order=1)
echt_Xf = echt.fit_transform(samples_orig)
echt_phase = np.angle(echt_Xf)

hilbert_Xf = hilbert(samples_orig)
hilbert_phase = np.angle(hilbert_Xf)

# Plot samples
plt.figure(figsize=(10, 5))
plt.plot(samples_orig, label='Original')
plt.plot(samples_phase, '-.', label='cecht online Phase')
plt.plot(samples_real, label='cecht online Real part')
plt.plot(cecht_phase, ':', label='cecHT offline Phase')
plt.plot(echt_phase, '--', label='ecHT offline Phase')
plt.plot(hilbert_phase, label='Hilbert offline Phase')

plt.xlabel('Message Index')
plt.ylabel('Sample Value')
plt.grid()
plt.legend()
plt.savefig('samples_comparison.png', dpi=300)

plt.show()