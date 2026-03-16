## Install requirements
```bash
pip install -r requirements.txt
sudo apt-get install libzmq3-dev
sudo apt install gcc-14 g++-14

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
# or in my case i needed to use this:
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
make

# Add the installation path in your $PATH if not already the case
export PATH="$PWD/build/falcon:$PATH"

# Allow falcon to more finely control CPU core utilization.
sudo setcap 'cap_sys_nice=pe' ./build/falcon/falcon 

# Check if installation worked
falcon --help
```
## Start Benchmark
```bash
python3 c_benchmark.py --overwrite
```
