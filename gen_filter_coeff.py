import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, freqz_sos, bessel 
from scipy import fftpack
import os

def gen_filter_ecHT(filter_params, output_folder):

    sos_outputs = []

    N, low_cutoff, high_cutoff, fs, length, btype = filter_params[0]

    if length is not None:
        # Store frequency response of bandpass filter (for PhaseEstimator)
        length = fftpack.next_fast_len(length)

    filename = f"{N}_{low_cutoff:.2f}_{high_cutoff:.2f}_{fs}{'_' + str(length) if length is not None else ''}.txt"
    output_path = f"{output_folder}/{filename}"
    if os.path.exists(output_path):
        return

    print(f"Generating coefficients for {low_cutoff:.2f}-{high_cutoff:.2f}Hz")
    
    if low_cutoff == 0:
        sos = butter(N=N, Wn=high_cutoff / (fs / 2), btype="low", output="sos")
    else:
        Wn = [low_cutoff / (fs / 2), high_cutoff / (fs / 2)]
        sos = butter(N=N, Wn=Wn, btype=btype, output="sos")

    sos_outputs.append(sos)

    if length is not None:
        # Store frequency response of bandpass filter (for PhaseEstimator)
        filt_freq = np.fft.fftfreq(length, d=1 / fs)
        _, H = freqz_sos(sos, worN=filt_freq, fs=fs)
        coeffs = H[:, None]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("##\n")
            f.write("# type = frequency response\n")
            f.write(f"# description = Frequency response of {btype} filter {low_cutoff:.2f}-{high_cutoff:.2f}Hz @ {fs}Hz ({length} samples)\n")
            f.write("# format = text\n")
            f.write("##\n")

            for i in range(len(filt_freq)):
                f.write(f"{H[i].real:.18g} {H[i].imag:.18g}\n")

def gen_filter(filter_params, fs, output_folder):

    sos_outputs = []

    filename = f"global_filter_{fs}.txt"
    output_path = f"{output_folder}/{filename}"

    print(f"Generating coefficients for global filter")

    for N, low_cutoff, high_cutoff, btype in filter_params:
        print(f"Adding filter: {N}, {btype}, {low_cutoff:.2f}-{high_cutoff:.2f}Hz")

        if low_cutoff == 0:
            sos = butter(N=N, Wn=high_cutoff / (fs / 2), btype="low", output="sos")
        else:
            Wn = [low_cutoff / (fs / 2), high_cutoff / (fs / 2)]
            sos = butter(N=N, Wn=Wn, btype=btype, output="sos")

        sos_outputs.append(sos)

    global_filter = np.vstack(sos_outputs)

    # Store filter coefficients (for MultiChannelFilter)
    description = (f"Global filter @ {fs}Hz")

    with open(output_path, "w", encoding="utf-8") as f:
        # Header format expected by parse_file_header in lib/dsp/filter.cpp.
        f.write("##\n")
        f.write("# type = sos\n")
        f.write(f"# description = {description}\n")
        f.write("# format = text\n")
        f.write("##\n")

        # SOSFilter::FromStream expects: gain on first numeric line,
        # then flattened SOS rows as whitespace-separated values.
        f.write("1.0\n")
        for row in global_filter:
            f.write(" ".join(f"{coef:.18g}" for coef in row) + "\n")

    # Store frequency response of bandpass filter (for Phasedrift compensation in PhaseEstimator)
    freqs = np.arange(0, high_cutoff, 0.1)  # from 0 to Nyquist in 0.1 Hz steps (because IAF can change in 0.1 Hz steps)
    w, H = freqz_sos(global_filter, worN=freqs, fs=fs)
    phase = np.angle(H)  # phase shift in radians at each 0.1 Hz step

    output_path = os.path.join(output_folder, filename.replace(".txt", "_phase.txt"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("##\n")
        f.write("# type = phase shift\n")
        f.write(f"# description = Phase Shift of global filter @ {fs}Hz (0.1 Hz steps)\n")
        f.write("# format = text\n")
        f.write("##\n")

        for i in range(len(freqs)):
            f.write(f"{phase[i]:.18g}\n")

    return

def plot_filter_response(sos, fs):
    freqs = np.arange(0, fs/2, 0.1)
    f, H = freqz_sos(sos, worN=freqs, fs=fs)

    magnitude_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase_rad = np.angle(H, deg=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.set_facecolor("white")

    ax1.plot(f, magnitude_db, linewidth=2)
    ax1.set_title("Filter Frequency Response")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, alpha=0.8)

    ax2.plot(f, phase_rad, 'o-',linewidth=2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, alpha=0.8)

    plt.tight_layout()


if __name__ == "__main__":
    # Bandpass
    N = 1
    low_cutoff = 2.5
    high_cutoff = 35
    fs = 10000

    sos1 = butter(
        N=N,
        Wn=[low_cutoff / (fs / 2), high_cutoff / (fs / 2)],
        btype="bandpass",
        output="sos",
    )

    sos2 = butter(
        N=N,
        Wn=[48 / (fs / 2), 52 / (fs / 2)],
        btype="bandstop",
        output="sos",
    )
    
    plot_filter_response(np.vstack([sos1, sos2]), fs)

    # # Bandstop
    # N = 1
    # low_cutoff = 48
    # high_cutoff = 52
    # fs = 10000

    # sos = butter(
    #     N=N,
    #     Wn=[low_cutoff / (fs / 2), high_cutoff / (fs / 2)],
    #     btype="bandstop",
    #     output="sos",
    # )
    
    # plot_filter_response(sos, fs)

    plt.show()