import math
import socket
import struct
import time
import zmq
import numpy as np

UDP_IP = "127.0.0.1"
UDP_PORT = 25000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
token = 0xAABBCCDD
trigger_bits = 0

# Signal parameters
amplitude = 3.0
frequency = 10
sample_rate = 1000.0

dt = 1.0 / sample_rate
t = 0.0

# Modulation parameters
Modulation_type = "None"   # amplitude, phase or None
mod_ampl = 1
mod_freq = 0.05

print(f"Sending structured packets to {UDP_IP}:{UDP_PORT}")

# ZeroMQ 
serialization_rate_hz = 100.0
serialization_interval = sample_rate / serialization_rate_hz
socket = zmq.Context().socket(zmq.PUB)
socket.bind("tcp://localhost:5555")
print("ZeroMQ publisher bound to tcp://localhost:5555")

sample_counter = 0

storage = []
try:
    while True:
        # --- AUX (8 floats) ---
        aux = [0.0] * 8

        # --- EEG (32 floats) ---
        eeg = []

        carrier =  amplitude * math.cos(2.0 * math.pi * frequency * t)

        if Modulation_type != "None":
            if Modulation_type == "amplitude": 
                value = carrier + mod_ampl/2 * (
                                                math.cos(2.0 * math.pi * (frequency + mod_freq) * t) +
                                                math.cos(2.0 * math.pi * (frequency - mod_freq) * t)
                                            )
                theta = 2.0 * math.pi * frequency * t
                inst_freq = frequency

            elif Modulation_type == "phase":
                value = amplitude * math.cos(2.0 * math.pi * frequency * t + (mod_ampl/mod_freq) * math.sin(2.0 * math.pi * mod_freq * t))
                theta = 2.0 * math.pi * frequency * t + (mod_ampl/mod_freq) * math.sin(2.0 * math.pi * mod_freq * t)
                inst_freq = frequency + mod_ampl * math.cos(2.0 * math.pi * mod_freq * t)
        else:
            value = carrier
            theta = 2.0 * math.pi * frequency * t
            inst_freq = frequency

        current_phase = ((theta + math.pi) % (2.0 * math.pi)) - math.pi # wrap to [-pi, pi]

        for ch in range(32):
            eeg.append(value)

        storage.append((t, value, current_phase, inst_freq))

        # eeg[9] *= 10    # Make channel 9 stand out for testing

        
        # print(f"Time: {t:.3f}s, Sample: {sample_counter}, inst_freq: {inst_freq:.3f}, Phase: {current_phase:.3f}")  # Print time, sample count, first EEG channel and current phase

        # --- Pack ---
        packet_udp = struct.pack(
            "<III"      # token, sample_counter, trigger_bits
            "8f"        # aux
            "32f",      # eeg
            token,
            sample_counter,
            trigger_bits,
            *aux,
            *eeg
        )
        sock.sendto(packet_udp, (UDP_IP, UDP_PORT))

        # ZMQ only sends sample counter and current phase
        if sample_counter % serialization_interval == 0:
            packet_zmq = struct.pack(
                "<If",      # sample_counter, current_phase
                sample_counter,
                current_phase
            )            
            socket.send(packet_zmq)

        # Advance time + counter
        t += dt
        sample_counter += 1

        # Maintain approximate real-time rate
        time.sleep(dt)
        
except KeyboardInterrupt:
    np.save("simulated_signal.npy", np.array(storage, dtype=[("time", "f4"), ("value", "f4"), ("phase", "f4"), ("inst_freq", "f4")]))