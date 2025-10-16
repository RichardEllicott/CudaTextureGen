#!/bin/bash
set -e # Exit on error

export CPLUS_INCLUDE_PATH=/usr/include/python3.12

# might need to beware of line endings (although i set vs.code now to use \n)
# dos2unix BUILD.sh

BUILD_DIR="build/linux"
mkdir -p $BUILD_DIR
cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C "$BUILD_DIR"






