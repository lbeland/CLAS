import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
samples = pd.read_csv('rt_c_results/10_1_samples.csv', header=0)

# Plot samples
plt.figure(figsize=(10, 5))
plt.plot(samples["Orig"].values, label='Original Samples')
plt.plot(samples["Phase"].values, label='Phase Samples')
plt.plot(samples["Real"].values, label='Real Samples')

plt.title('Samples Received by Consumer')
plt.xlabel('Message Index')
plt.ylabel('Sample Value')
plt.grid()
plt.legend()
plt.savefig('samples_comparison.png', dpi=300)