#!/bin/sh

src="./build/linux/src/python/cuda_texture_gen.cpython-312-x86_64-linux-gnu.so"
dst="./lib/cuda_texture_gen/cuda_texture_gen.cpython-312-x86_64-linux-gnu.so"
cp "$src" "$dst"

cd ./lib/cuda_texture_gen
stubgen -m cuda_texture_gen --include-private -o . # ends up here
cd ../..
