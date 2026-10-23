# z_extra — helper scripts

Small tools written while developing and testing CLAS. None of them are part of a normal CLAS
run. Run everything from the repo root.

## Testing CLAS without hardware

| file | what |
|------|------|
| [`simulate_EEG.py`](simulate_EEG.py) | sends a synthetic (optionally AM/PM-modulated) cosine as UDP packets to `UDPSource` (port 25000) and saves the ground truth to `simulated_signal.npy` on exit. See "Simulate a UDP sender" in the main README |
| [`live_plot_zmq.py`](live_plot_zmq.py) | **early draft, unmaintained:** live plot of CLAS streams over ZeroMQ. Probably no longer works (no current graph has a ZMQ serializer) |

## Graphs and setup

| file | what |
|------|------|
| [`plot_processor_flowchart.py`](plot_processor_flowchart.py) | renders a graph YAML as a flowchart (needs `graphviz`). By default writes an `.svg` next to the YAML; `-o`/`--format` to change |
| [`precompute_wisdom.cpp`](precompute_wisdom.cpp) | precomputes FFTW wisdom for all FFT sizes used by `PhaseEstimation` (f0 5–18 Hz) and `FrequencyEstimation`, so they never plan at runtime. Settings are hardcoded to match the graphs (fs = 10 kHz, `window_size_sec` = 10) |

```bash
python3 z_extra/plot_processor_flowchart.py resources/graphs/TurboLinkCLAS.yaml

g++ -O2 -o z_extra/precompute_wisdom z_extra/precompute_wisdom.cpp -lfftw3
./z_extra/precompute_wisdom      # updates resources/fft_wisdom/fftw_wisdom.txt
```

## Hardware tests

| file | what |
|------|------|
| [`TurbolinkMeas/`](TurbolinkMeas/) | measures TurboLink UDP packet timing / jitter (`measurement.cpp`) and plots the packet delay variation (`plotMeas.py`). See [TurbolinkMeas/meas.md](TurbolinkMeas/meas.md) and the TurboLink troubleshooting section in the main README |
| [`generate_arb_NGU411.py`](generate_arb_NGU411.py) | writes sine-wave ARB `.csv` files for the Rohde & Schwarz NGU411, with exact step durations so the frequency is reproduced without rounding |
| [`audio_gen.cpp`](audio_gen.cpp) | standalone ALSA test: streams silence and plays a pink-noise burst whenever you press ENTER |
| [`plot_burst_buffer.py`](plot_burst_buffer.py) | plots waveform and spectrum of StimControl's precomputed noise burst. Needs the debug dump `results/burst_buffer_debug.txt`, enabled in `StimControl.cpp` by setting `burst_buffer_dumped = false` |

```bash
python3 z_extra/generate_arb_NGU411.py 10.5 -a 0.01 -o 10mV_10.5Hz.csv   # frequency (Hz), amplitude (V), output

g++ -O2 -I. -o audio_gen z_extra/audio_gen.cpp -lasound -lpthread
./audio_gen
```
