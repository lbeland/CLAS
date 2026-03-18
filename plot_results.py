import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
samples = pd.read_csv('rt_c_results/1_1_samples.csv', header=None).values.flatten()

# Plot samples
plt.figure(figsize=(10, 5))
plt.plot(samples, marker='o')
plt.title('Samples Received by Consumer')
plt.xlabel('Message Index')
plt.ylabel('Sample Value')
plt.grid()
plt.show()