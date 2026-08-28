## Install requirements
```bash
pip install -r requirements.txt
sudo apt-get install libzmq3-dev
sudo apt install libboost-dev   # for boost ringbuffer support
sudo apt install gcc-14 g++-14

mkdir build
make clean-cmake

cmake .. -B build/debug -DCMAKE_BUILD_TYPE=Debug
# or in my case i needed to use this:
cmake -B build/debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
cmake --build build/debug -- -j$(nproc) 

# or for release:
cmake -B build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
cmake --build build/release -- -j$(nproc)

# Add the installation path in your $PATH if not already the case
export PATH="$PWD/build/falcon:$PATH"

# Check if installation worked
falcon --help
```

## FFTW Installation
- fftw-3.3.10 folder holds the library
- has to be compiled with 
```bash
./configure
make
make install
```
- there exist several compile flags for specializations (float or double precision, threads enabled etc. https://www.fftw.org/fftw2_doc/fftw_6.html )
- check later if some of these could lead to faster execution

./configure --enable-float --enable-threads --enable-openmp --enable-sse2 --enable-avx --enable-avx2 CFLAGS="-O3 -march=native -mtune=native"
(used on WSL Windows)

## Start Simulation
```bash
python3 clas.py
```

## Generate Flowchart of graph
python3 plot_processor_flowchart.py resources/graphs/SimulateCLAS.yaml

## Disable power save of audio card
Create config file
```bash
echo "options snd_hda_intel power_save=0" | sudo tee /etc/modprobe.d/audio_disable_powersave.conf
```

## Isolate CPU Cores and keep busy spinning without interruption by OS
sudo sysctl kernel.sched_rt_runtime_us=1000000
sudo sysctl kernel.sched_rt_period_us=1000000 


in /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash isolcpus=6,7,8,9,10,18,19,20,21,22 nohz_full=6,7,8,9,10,18,19,20,21,22 rcu_nocbs=6,7,8,9,10,18,19,20,21,22 irqaffinity=0-5,11,12-17,23

## After executing the programm the sound on the PC may not work anymore because the program took over the sound card. To restart normal Pulsewire:
systemctl --user restart pipewire.service

## Caution when connecting the TurboLink via an USB Ethernet adapter
- Symptom: samples arrive in bursts rather than with the nominal inter-sample interval (you can check this by executing the script [measurement.cpp](TurbolinkMeas/measurement.cpp) followed by [plotMeas.py](TurbolinkMeas/plotMeas.py) -> the PDV histogram shall show a single uniform distribution instead of having multiple maxima)
- Cause: USB Ethernet adapters (interface names like `enx<mac>`) can have NIC interrupt coalescing (`rx-usecs`) set very high by default, so incoming packets are batched and delivered to the kernel/app in bursts instead of individually.
- Check current setting (replace `<iface>` with your adapter, e.g. from `ip -brief link show`):
```bash
sudo ethtool -c <iface>
```
- Fix (not persistent, resets on reboot/replug):
```bash
sudo ethtool -C <iface> rx-usecs 0
```
- Make it persistent by adding a udev rule that reapplies the setting whenever the adapter is connected (the `enx<mac>` name is stable per device):
```bash
# /etc/udev/rules.d/71-usb-eth-coalesce.rules
ACTION=="add", SUBSYSTEM=="net", KERNEL=="<iface>", RUN+="/usr/sbin/ethtool -C %k rx-usecs 0"
```
```bash
sudo udevadm control --reload-rules
```
- Note: some USB Ethernet drivers don't actually implement coalescing and may silently ignore the setting — re-run `ethtool -c <iface>` after setting it to confirm the value changed.