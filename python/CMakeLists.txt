# # Add nanobind as a subdirectory (assumes it's cloned into python/nanobind)
# add_subdirectory(nanobind)

# # Define the Python module target (internal CMake name: cuda_hello_py)
# # This compiles bindings.cpp into a shared library for Python
# nanobind_add_module(cuda_hello_py bindings.cpp)

# # Rename the output .so file to "cuda_hello.so"
# # This ensures Python can import it as `import cuda_hello`
# # The C++ NB_MODULE name must also be cuda_hello to match
# set_target_properties(cuda_hello_py PROPERTIES OUTPUT_NAME "cuda_hello")





add_subdirectory(nanobind)

nanobind_add_module(cuda_hello_py bindings.cpp)
target_link_libraries(cuda_hello_py PRIVATE cuda_core)
set_target_properties(cuda_hello_py PROPERTIES OUTPUT_NAME "cuda_hello")
