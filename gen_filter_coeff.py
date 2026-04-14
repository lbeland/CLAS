import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, freqz_sos
import os

def gen_bandpass(N, low_cutoff, high_cutoff, fs, length, output_folder):

    filename = f"{N}_{low_cutoff:.1f}_{high_cutoff:.1f}_{fs}{'_' + str(length) if length is not None else ''}.txt"
    output_path = f"{output_folder}/{filename}"
    if os.path.exists(output_path):
        return

    print(f"Generating coefficients for {low_cutoff:.2f}-{high_cutoff:.2f}Hz")
    
    if low_cutoff == 0:
        sos = butter(N=N, Wn=high_cutoff / (fs / 2), btype="low", output="sos")
    else:
        Wn = [low_cutoff / (fs / 2), high_cutoff / (fs / 2)]
        sos = butter(N=N, Wn=Wn, btype="band", output="sos")

    if length is not None:
        # Store frequency response of bandpass filter (for PhaseEstimator)
        filt_freq = np.fft.fftfreq(length, d=1 / fs)
        _, H_center = freqz_sos(sos, worN=filt_freq, fs=fs)
        coeffs = H_center[:, None]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("##\n")
            f.write("# type = frequency response\n")
            f.write(f"# description = Frequency response of bandpass filter {low_cutoff:.2f}-{high_cutoff:.2f}Hz @ {fs}Hz ({length} samples)\n")
            f.write("# format = text\n")
            f.write("##\n")

            for i in range(len(filt_freq)):
                f.write(f"{H_center[i].real:.18g} {H_center[i].imag:.18g}\n")
    else:
        # Store filter coefficients (for MultiChannelFilter)
        description = (
            f"Bandpass SOS {low_cutoff:.2f}-{high_cutoff:.2f}Hz @ {fs}Hz"
        )

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
            for row in sos:
                f.write(" ".join(f"{coef:.18g}" for coef in row) + "\n")

    return

def plot_filter_response(sos, fs):
    f, H = freqz_sos(sos, worN=4096, fs=fs)

    magnitude_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase_rad = np.unwrap(np.angle(H))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.set_facecolor("white")

    ax1.plot(f, magnitude_db, linewidth=2)
    ax1.set_title("Bandpass Filter Frequency Response")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(f, phase_rad, linewidth=2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (rad)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    N = 1
    low_cutoff = 1
    high_cutoff = 50
    fs = 1000

    gen_bandpass(N, low_cutoff, high_cutoff, fs, length=None, output_folder=".")

    if low_cutoff == 0:
        sos = butter(N=N, Wn=high_cutoff / (fs / 2), btype="low", output="sos")
    else:
        sos = butter(
            N=N,
            Wn=[low_cutoff / (fs / 2), high_cutoff / (fs / 2)],
            btype="band",
            output="sos",
        )

    plot_filter_response(sos, fs)