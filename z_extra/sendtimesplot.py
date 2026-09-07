import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = "/home/linda/Documents/CLAS/build/debug/falcon/_last_run/"
dir ="_last_run/"

df = pd.read_csv(f"{dir}/SimulatedSource/send_times.csv", header=None)
send_times = df[0].values

high = np.where(send_times > 10000)[0]
print(f"High send times at indices: {high}, values: {send_times[high]}")

# df = pd.read_csv(f"{dir}/BenchSink/recv_times.csv", header=None)
# recv_times = df[0].values

# df = pd.read_csv(f"{dir}/BenchSink/process_times.csv", header=None)
# process_times = df[0].values

# df = pd.read_csv(f"{dir}/BenchSink2/recv_times.csv", header=None)
# recv_times2 = df[0].values

# df = pd.read_csv(f"{dir}/BenchSink2/process_times.csv", header=None)
# process_times2 = df[0].values

plt.plot(send_times, alpha=0.7, label="SimulatedSource send times")
# plt.plot(recv_times, alpha=0.7, label="BenchSink receive times")
# plt.plot(process_times, alpha=0.7, label="BenchSink process times")
# plt.plot(recv_times2, alpha=0.7, label="BenchSink2 receive times")
# plt.plot(process_times2, alpha=0.7, label="BenchSink2 process times")

plt.xlabel("Message index")
plt.ylabel("Time (s)")
plt.title("SimulatedSource Send Times")
plt.legend()
plt.show()
