#!/bin/bash
set -e

export CPLUS_INCLUDE_PATH=/usr/include/python3.12
# export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

mkdir -p build
cd build

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja

# Run Python test from correct directory
cd python
python3 -c "import cuda_hello; print(cuda_hello.hello())"



