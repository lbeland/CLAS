import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meegkit.phase import ECHT

# Load data
samples = pd.read_csv('rt_c_results/10_1_samples.csv', header=0)

orig = samples["Orig"].values
f0 = 10
filt_BW = f0 / 2
l_freq = f0 - filt_BW / 2
h_freq = f0 + filt_BW / 2

echt = ECHT(l_freq, h_freq, 10000, filt_order=1)
Xf = echt.fit_transform(orig)
phase = np.angle(Xf)

# plt.plot(echt.coef_)
# plt.show()

# Plot samples
plt.figure(figsize=(10, 5))
plt.plot(samples["Orig"].values, label='Original')
plt.plot(samples["Phase"].values, label='Falcon Phase ')
plt.plot(samples["Real"].values, label='FalconReal Samples')
plt.plot(phase, label='ECHT Phase')

plt.title('Samples Received by Consumer')
plt.xlabel('Message Index')
plt.ylabel('Sample Value')
plt.grid()
plt.legend()
plt.savefig('samples_comparison.png', dpi=300)

plt.show()