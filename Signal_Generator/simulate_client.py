import math
import socket
import struct
import time
import zmq
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import numpy as np

def ampl_mod(t, A_c, f_c, A_m, f_m):
    w_c = 2.0 * math.pi * f_c
    w_m = 2.0 * math.pi * f_m
    value = (A_c + A_m * math.cos(w_m * t)) * math.cos(w_c * t)
    theta = w_c * t
    inst_freq = w_c / (2.0 * math.pi)

    return value, theta, inst_freq

def angle_mod(t, A_c, f_c, A_m, f_m):           
    w_c = 2.0 * math.pi * f_c
    w_m = 2.0 * math.pi * f_m
    value = A_c * math.cos(w_c * t + (A_m/f_m) * math.sin(w_m * t))
    theta = w_c * t + (A_m/f_m) * math.sin(w_m * t)
    inst_freq = f_c + A_m * math.cos(w_m * t)

    return value, theta, inst_freq

def client_simulate(n_samples=-1):
    UDP_IP = "127.0.0.1"
    UDP_PORT = 25000
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    token = 0xAABBCCDD
    trigger_bits = 0

    # Signal parameters
    carrier_signal = {'amplitude': 3.0, 'frequency': 8}

    sample_rate = 10000.0
    dt = 1.0 / sample_rate
    t = 0.0

    # Modulation parameters
    Modulation_type = "amplitude"   # amplitude, phase or None
    mod_signal = {'amplitude': 1.0, 'frequency': 0.1}

    print(f"Sending structured packets to {UDP_IP}:{UDP_PORT}")

    # ZeroMQ 
    serialization_rate_hz = 100.0
    serialization_interval = sample_rate / serialization_rate_hz
    zmq_socket = zmq.Context().socket(zmq.PUB)
    zmq_socket.bind("tcp://localhost:5555")
    print("ZeroMQ publisher bound to tcp://localhost:5555")

    sample_counter = 0

    storage = []
    try:

        while True:
            if n_samples > 0 and sample_counter >= n_samples:
                break

            # --- AUX (8 floats) ---
            aux = [0.0] * 8

            # --- EEG (32 floats) ---
            eeg = []

            if Modulation_type != "None":
                if Modulation_type == "amplitude": 
                    value, theta, inst_freq = ampl_mod(t, carrier_signal["amplitude"], carrier_signal["frequency"], mod_signal["amplitude"], mod_signal["frequency"])

                elif Modulation_type == "phase":
                    value, theta, inst_freq = angle_mod(t, carrier_signal["amplitude"], carrier_signal["frequency"], mod_signal["amplitude"], mod_signal["frequency"])

            else:
                value = carrier_signal["amplitude"] * math.cos(2.0 * math.pi * carrier_signal["frequency"] * t)
                theta = 2.0 * math.pi * carrier_signal["frequency"] * t
                inst_freq = carrier_signal["frequency"]

            for ch in range(32):
                eeg.append(value)

            storage.append((t, value, theta, inst_freq))

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
            udp_sock.sendto(packet_udp, (UDP_IP, UDP_PORT))

            # ZMQ only sends sample counter and current phase
            if sample_counter % serialization_interval == 0:
                packet_zmq = struct.pack(
                    "<If",      # sample_counter, current_phase
                    sample_counter,
                    theta
                )            
                zmq_socket.send(packet_zmq)

            # Advance time + counter
            t += dt
            sample_counter += 1

            # Maintain approximate real-time rate
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        np.save("simulated_signal.npy", np.array(storage, dtype=[("time", "f4"), ("value", "f4"), ("phase", "f4"), ("inst_freq", "f4")]))


def main():
    client_simulate(-1)
    # loaded = np.load("simulated_signal.npy")

    # fig, axes = plt.subplots(3,1, figsize=(10,7), sharex=True)
    # axes[0].plot(loaded["time"], loaded["value"], label="Signal Value")
    # axes[1].plot(loaded["time"], loaded["phase"], label="Phase")
    # axes[1].plot(loaded["time"], np.unwrap(np.angle(hilbert(loaded["value"]))), label="Hilbert Phase", linestyle="--")
    # axes[2].plot(loaded["time"], loaded["inst_freq"], label="Instantaneous Frequency")
    # axes[0].set_ylabel("Value")
    # axes[1].set_ylabel("Phase")
    # axes[2].set_ylabel("Instantaneous Frequency")
    # axes[2].set_xlabel("Time")
    # axes[0].legend()
    # axes[1].legend()
    # axes[2].legend()
    # plt.show()

if __name__ == "__main__":
    main()