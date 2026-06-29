import matplotlib.pyplot as plt
import numpy as np




sigma_min = 0.042   # Hz
sigma_max = 2   # Hz
sigma = np.linspace(sigma_min,sigma_max, 100)[:,None]
snr = np.linspace(0.3,100, 100)[None,:] # SNR range from -40dB to 40dB

R = sigma**2 / (snr)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) #, "zscale": "log"})
ax.plot_surface(20*np.log10(snr), sigma, R, cmap="viridis")
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Sigma (Hz)')
ax.set_zlabel('R')
ax.set_title('Variation of R with SNR and Sigma')
plt.show()



## CRB with outlier probability q
# from scipy.special import gammaln
# from scipy.special import factorial
# fs = 4000
# T = 1/fs
# N = 16
# SNR = 10 **(1 / 10)
# m = np.arange(2, N+1, 1)
# log_binom = gammaln(N+1) - gammaln(N-m+1) - gammaln(m+1)
# log_gauss = -N * SNR * (m-1) / m
# signs     = (-1.0)**m
# terms = signs * np.exp(log_binom + log_gauss)
# q     = np.sum(terms) / N
# print(q)
## Example SNR values in dB
# snr_db = np.linspace(-20, 30, 100)  
# snr = 10 ** (snr_db / 20)  # Convert to linear scale
# R = 6 / (4*pow(np.pi,2)*snr*pow(T,2)*N*(pow(N,2)-1))
# plt.semilogy(snr_db, R, 'o')
# plt.xlabel('SNR (dB)')
# plt.ylabel('R')

# plt.title('Variation of R with SNR')
# plt.show()
