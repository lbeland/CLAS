## Install requirements
```bash
pip install -r requirements.txt
sudo apt-get install libzmq3-dev
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

## Savitzky Golay Installation for IAF Estimation
- gram_savitzky_golay folder holds the library
- has to be compiled with 
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
sudo make install
```


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