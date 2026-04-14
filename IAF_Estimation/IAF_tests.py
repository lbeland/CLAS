import numpy as np
import matplotlib.pyplot as plt
import time
from specparam import SpectralGroupModel
from specparam.bands import Bands
from specparam.data.periodic import get_band_peak_group
from specparam.plts.spectra import plot_spectra

np.random.seed(0)  # For reproducibility


def main():
    fs = 10000.0
    signal_length = int(4*fs)  # 2 seconds of data
    window_length = int(2*fs) # 1 seconds window
    fft_length = 2**17 

    f0 = 10
    print(f"FFT length: {fft_length}, Frequency resolution: {fs/fft_length:.4f} Hz")

    t = np.arange(signal_length) / fs
    x = np.sin(2 * np.pi * f0 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.rand(signal_length)  # one cycle every 1000 samples

    for start in range(1, signal_length - window_length, 100):
        window = x[start:start + window_length] #* np.hanning(window_length)  # Apply Hanning window to reduce spectral leakage
        freq = np.fft.rfft(window, fft_length)
        freq_bins = np.fft.rfftfreq(fft_length, 1/fs)

        max_freq_foof = foof(freq, freq_bins)
        max_freq_stupid = stupid_max(freq, freq_bins)
        print(f"FOOF max frequency: {max_freq_foof:.4f} Hz")
        print(f"Stupid max frequency: {max_freq_stupid:.4f} Hz")
        # if max_freq != f0:
        #     print(f"Unexpected max frequency: {max_freq} Hz at iteration {start}")
        # print(f"Maximum frequency: {max_freq} Hz")
        break

def stupid_max(freq, freq_bins):
    mask = (freq_bins >= 7) & (freq_bins <= 14)
    alpha_bins = np.where(mask)[0]                  # absolute bin indices
    rel_idx = np.argmax(np.abs(freq[alpha_bins]))   # index within alpha_bins
    max_bin = alpha_bins[rel_idx]                   # convert to absolute index
    return freq_bins[max_bin]

def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data

def foof(freq, freq_bins):
    # Initialize a SpectralGroupModel object, with desired settings
    fg = SpectralGroupModel(peak_width_limits=[1, 6], min_peak_height=0.15,
                    peak_threshold=2., max_n_peaks=6, verbose=False)

    # Define the frequency range to fit
    freq_range = [1, 30]

    ###################################################################################################

    # Fit the power spectrum model across all channels
    fg.fit(freq_bins, np.abs(freq[None,:],dtype=np.float64), freq_range)

    ###################################################################################################

    # Check the overall results of the group fits
    # fg.plot()

    # Define frequency bands of interest
    bands = Bands({'theta': [3, 7],
                'alpha': [7, 14],
                'beta': [15, 30]})

    ###################################################################################################

    # Extract alpha peaks
    alphas = get_band_peak_group(fg, bands['alpha'])

    # Extract the power values from the detected peaks
    alpha_pw = alphas[:, 1]

    # Define frequency bands of interest
    bands = Bands({'theta': [3, 7],
                'alpha': [7, 14],
                'beta': [15, 30]})

    ###################################################################################################

    # Extract alpha peaks
    alphas = get_band_peak_group(fg, bands['alpha'])
    betas = get_band_peak_group(fg, bands['beta'])
    thetas = get_band_peak_group(fg, bands['theta'])

    # Extract the power values from the detected peaks
    alpha_pw = alphas[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ind, (label, band_def) in enumerate(bands):

        # Get the power values across channels for the current band
        band_power = check_nans(get_band_peak_group(fg, band_def)[:, 1])

        # Extracted and plot the power spectrum model with the most band power
        fg.get_model(np.argmax(band_power)).plot(ax=axes[ind], add_legend=False)

        # Set some plot aesthetics & plot title
        axes[ind].yaxis.set_ticklabels([])
        axes[ind].set_title('biggest ' + label + ' peak', {'fontsize' : 16})

    plt.savefig('IAF_Estimation/test.png')
    plt.show()
    return alphas[0,0]

def parabolic_max(freq, freq_bins):
    # This is a placeholder for a parabolic interpolation method to refine frequency estimation
    return stupid_max(freq, freq_bins)

if __name__ == "__main__":
    main()