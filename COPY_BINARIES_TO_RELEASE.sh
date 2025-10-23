#!/bin/bash

src="./build/linux/python/cuda_texture_gen.cpython-312-x86_64-linux-gnu.so"
dst="./bin/cuda_texture_gen.cpython-312-x86_64-linux-gnu.so"

# Ensure destination folder exists
mkdir -p "$(dirname "$dst")"

# Check if destination file exists
if [ -f "$dst" ]; then
    read -p "File already exists at $dst. Overwrite? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "Copy aborted."
        exit 1
    fi
fi

# Copy the file
cp "$src" "$dst"
echo "File copied to $dst."