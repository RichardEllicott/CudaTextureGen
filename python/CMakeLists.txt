







# Include the nanobind CMake logic (usually from a submodule or local folder)
# This makes the `nanobind_add_module()` function available
add_subdirectory(nanobind)

# Define a Python extension module named `cuda_hello_py`
# This compiles `bindings.cpp` and wraps it as a `.pyd` (Windows) or `.so` (Linux/macOS)
nanobind_add_module(cuda_hello_py bindings.cpp)

# Link the Python module against your CUDA static library
# This allows Python to call into CUDA kernels or host-side logic defined in `cuda_core`
target_link_libraries(cuda_hello_py PRIVATE cuda_core)

# Rename the output file to `cuda_hello` so you can `import cuda_hello` in Python
set_target_properties(cuda_hello_py PROPERTIES OUTPUT_NAME "cuda_hello")


