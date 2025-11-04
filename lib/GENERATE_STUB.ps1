# generate a stub, allows vs.code to explore the python library
# pip install mypy

cd cuda_texture_gen
stubgen -m cuda_texture_gen --include-private -o . # ends up here
cd ..
