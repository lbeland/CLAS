## Comparison of Sliding DFT with FFTW
- in [Python](./testSDFT.py) and [C++](./sdft/cpp/examples/testsdft.cpp)

to run C++ code: 
```bash
cd SlidingDFT/sdft/cpp
cmake -S . -B build     # once
cmake --build build --parallel  # after code update
./build/sdft-example-testsdft-compare
```

