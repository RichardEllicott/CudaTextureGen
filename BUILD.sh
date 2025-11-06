#!/bin/sh
set -e # Exit on error
BUILD_DIR="build/linux"
BUILD_TYPE="Release" # Debug, Release, RelWithDebInfo

#
# dos2unix BUILD.sh
#
#

echo "🧙 Launching build..."

#region ✅ Checks
# ================================================================================================================================
echo "✅ Running checks..."
# --------------------------------------------------------------------------------------------------------------------------------
CHECK="cmake"
if command -v $CHECK >/dev/null 2>&1; then
    # echo "✔️ $CHECK found"
    :
else
    echo "⚠️ WARNING \"$CHECK\" not found"
fi
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
CHECK="ninja"
if command -v $CHECK >/dev/null 2>&1; then
    # echo "✔️ $CHECK found"
    :
else
    echo "⚠️ WARNING \"$CHECK\" not found"
fi
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
CHECK="sccache"
if command -v $CHECK >/dev/null 2>&1; then
    # echo "✔️ $CHECK found"
    :
else
    echo "⚠️ WARNING \"$CHECK\" not found"
fi
# --------------------------------------------------------------------------------------------------------------------------------

#endregion


#region 🔨 Build

echo "----------------------------------------------------------------"
echo "🔨 Build..."

export CPLUS_INCLUDE_PATH=/usr/include/python3.12

mkdir -p $BUILD_DIR
cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=$BUILD_TYPE
ninja -C "$BUILD_DIR"

echo "----------------------------------------------------------------"
echo "🎉 Build complete"


#endregion
