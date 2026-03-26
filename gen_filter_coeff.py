import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meegkit.phase import ECHT

# Load data
samples = np.arange(2048)

f0 = 10
filt_BW = f0 / 2
l_freq = f0 - filt_BW / 2
h_freq = f0 + filt_BW / 2

echt = ECHT(l_freq, h_freq, 10000, filt_order=1)
Xf = echt.fit(samples)

plt.plot(echt.coef_)
plt.show()
np.savetxt('echt_coefficients.txt', np.vstack((np.real(echt.coef_[:,0]), np.imag(echt.coef_[:,0]))).T, delimiter=',')

