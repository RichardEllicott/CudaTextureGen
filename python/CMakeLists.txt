add_subdirectory(nanobind)
nanobind_add_module(cuda_hello_py bindings.cpp)
target_link_libraries(cuda_hello_py PRIVATE cuda_core)
set_target_properties(cuda_hello_py PROPERTIES OUTPUT_NAME "cuda_hello")
