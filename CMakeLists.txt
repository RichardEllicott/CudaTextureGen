cmake_minimum_required(VERSION 3.18)
project(cuda_hello LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Main CUDA target
add_executable(cuda_hello src/main.cu)

# Add Python bindings
add_subdirectory(python)
