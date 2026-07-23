## Run

```bash
g++ -o measurement measurement.cpp
sudo chrt -f 99 taskset -c 6 ./measurement 
```

## Observations
- TurboLink Website shows Sample Interval of Recorder (Sample rate that is set in Workspace of Recorder Programm)
- TurboLink DataClient Sample rate can only be changed by stopping the Monitoring in Recorder Programm, changing the fs in Website (has to be lower or equal to Recorder Worksapce fs) and starting monitoring again
- DataClient can only receive packets if Recorder is in Monitoring state


(venv) linda@hgn:~/Documents/CLAS/TurbolinkMeas$ sudo chrt -f 99 taskset -c 6 ./measurement 
Listening on UDP port 25000 

Received 1800000 packets in 180000.574ms.
Estimated fs: 9999.9681
Missed packets: 0

--- Inter-arrival period (legacy) ---
 Average receive period (us): 100.00037
 Estimated rate: 9999.9628 Hz
 Max receive period (us): 254.912, idx: 170026
 Std receive period (us): 15.299

--- Grid-based jitter (relative to sample_counter) ---
 Estimated true sample rate (Hz): 9999.9581 (nominal 10000.0)
 Drift slope (us per sample):     0.000419
 Jitter mean (us):  74.893
 Jitter std  (us):  14.236
 Jitter min  (us):  0.000 (by construction, = 0)
 Jitter max  (us):  223.690, idx: 170027
 Jitter p50  (us):  74.408
 Jitter p95  (us):  99.213
 Jitter p99  (us):  107.156
Per-packet jitter saved to jitter_10000.csv