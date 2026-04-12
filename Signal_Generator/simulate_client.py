import math
import socket
import struct
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 25000
Modulation_type = "angle"   # amplitude, angle or none
mod_ampl = 3
mod_freq = 0.5

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Signal parameters
amplitude = 1.0
frequency = 9
sample_rate = 100.0

dt = 1.0 / sample_rate
t = 0.0

sample_counter = 0
token = 0xAABBCCDD
trigger_bits = 0

print(f"Sending structured packets to {UDP_IP}:{UDP_PORT}")

while True:
    # --- AUX (8 floats) ---
    aux = [0.0] * 8

    # --- EEG (32 floats) ---
    eeg = []
    for ch in range(32):
        carrier =  amplitude * math.cos(2.0 * math.pi * frequency * t)

        if Modulation_type != "none":
            if Modulation_type == "amplitude": 
                value = carrier + mod_ampl/2 * (
                                                math.cos(2.0 * math.pi * (frequency + mod_freq) * t) +
                                                math.cos(2.0 * math.pi * (frequency - mod_freq) * t)
                                            )
            elif Modulation_type == "angle":
                # freq_dev = frequency - mod_freq
                value = amplitude * math.cos(2.0 * math.pi * frequency * t + (mod_ampl/mod_freq) * math.sin(2.0 * math.pi * mod_freq * t))
        else:
            value = carrier

        eeg.append(value)

    eeg[9] *= 10    # Make channel 9 stand out for testing

    
    # print(f"Time: {t:.3f}s, Sample: {sample_counter}, EEG[0]: {eeg[0]:.3f}")

    # --- Pack ---
    packet = struct.pack(
        "<III"      # token, sample_counter, trigger_bits
        "8f"        # aux
        "32f",      # eeg
        token,
        sample_counter,
        trigger_bits,
        *aux,
        *eeg
    )

    sock.sendto(packet, (UDP_IP, UDP_PORT))

    # Advance time + counter
    t += dt
    sample_counter += 1

    # Maintain approximate real-time rate
    time.sleep(dt)