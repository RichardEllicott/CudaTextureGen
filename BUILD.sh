#!/bin/sh
set -e # Exit on error
BUILD_DIR="build/linux"
BUILD_TYPE="Release" # Debug, Release, RelWithDebInfo


#region ✅ Checks
# ================================================================================================================================
if command -v cmake >/dev/null 2>&1; then
    echo "cmake found"
else
    echo "cmake not found"
fi
# ================================================================================================================================
#endregion




export CPLUS_INCLUDE_PATH=/usr/include/python3.12

# might need to beware of line endings (although i set vs.code now to use \n)
# dos2unix BUILD.sh

mkdir -p $BUILD_DIR
cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=$BUILD_TYPE
ninja -C "$BUILD_DIR"