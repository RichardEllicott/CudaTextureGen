cmake_minimum_required(VERSION 3.18)
project(cuda_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)



# # Core CUDA library (no glob here can be safer)
# add_library(cuda_core STATIC
#     src/hello.cu
#     src/scale_kernel.cu
# )

# Auto-glob version (find all .cu files)
file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)
add_library(cuda_core STATIC ${CUDA_SOURCES})




target_include_directories(cuda_core PUBLIC include)
set_target_properties(cuda_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Add Python bindings
add_subdirectory(python)

# Optional: standalone executable for testing
add_executable(cuda_hello_exec src/hello_main.cpp)
target_link_libraries(cuda_hello_exec PRIVATE cuda_core)

# Optional: standalone executable for testing.... QUERY GPU
add_executable(cuda_query_gpu_exec src/query_gpu.cpp)
target_link_libraries(cuda_query_gpu_exec PRIVATE cuda_core)
