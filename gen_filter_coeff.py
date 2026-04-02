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


if __name__ == "__main__":
    # gen_bandpass(8, 12, 10000,2000)
    gen_bandpass(6,8,12,10000,output_folder=".")