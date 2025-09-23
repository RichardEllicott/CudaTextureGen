
# # Make Version
# mkdir -p build
# cd build
# cmake ..
# make
# cd ..

# Ninja Version
mkdir -p build
cd build
cmake .. -G Ninja
ninja
cd ..