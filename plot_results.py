import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meegkit.phase import ECHT
from scipy.signal import hilbert

# Load data
samples = pd.read_csv('rt_c_results/10_1_samples.csv', header=0)

orig = samples["Orig"].values
f0 = 10
filt_BW = f0 / 2
l_freq = f0 - filt_BW / 2
h_freq = f0 + filt_BW / 2

cecht = ECHT(l_freq, h_freq, 10000, filt_order=1, calibrate=True, f0=f0)
cecht_Xf = cecht.fit_transform(orig)
cecht_phase = np.angle(cecht_Xf)

echt = ECHT(l_freq, h_freq, 10000, filt_order=1)
echt_Xf = echt.fit_transform(orig)
echt_phase = np.angle(echt_Xf)

hilbert_Xf = hilbert(orig)
hilbert_phase = np.angle(hilbert_Xf)

# Plot samples
plt.figure(figsize=(10, 5))
plt.plot(samples["Orig"].values, label='Original')
plt.plot(samples["Phase"].values, '-.', label='cecht online Phase')
plt.plot(samples["Real"].values, label='cecht online Real part')
plt.plot(cecht_phase, ':', label='cecHT offline Phase')
plt.plot(echt_phase, '--', label='ecHT offline Phase')
plt.plot(hilbert_phase, label='Hilbert offline Phase')

plt.xlabel('Message Index')
plt.ylabel('Sample Value')
plt.grid()
plt.legend()
plt.savefig('samples_comparison.png', dpi=300)

plt.show()