#!/bin/bash
set -e # Exit on error

BUILD_DIR="build/linux"

# Only remove if it exists
if [ -d "$BUILD_DIR" ]; then
  rm -rf "$BUILD_DIR"
  echo "Removed $BUILD_DIR"
else
  echo "No build directory found at $BUILD_DIR"
fi
