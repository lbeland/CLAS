#!/bin/sh
set -e
cd build/debug
make -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
mkdir -p debug/debug
cp falcon/falcon debug/debug/
