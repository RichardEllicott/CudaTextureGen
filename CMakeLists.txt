cmake_minimum_required(VERSION 3.18)
project(cuda_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # will generate compile_commands.json for vs.code



# testing debug 🚧
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo -DDEBUG") # 🚧


# 🪟🪟🪟🪟 Python workaround for Windows
find_package(Python REQUIRED COMPONENTS Interpreter Development)

if(WIN32 AND NOT TARGET Python::Module)
    add_library(Python::Module UNKNOWN IMPORTED)
    set_target_properties(Python::Module PROPERTIES
        IMPORTED_LOCATION "${Python_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Python_INCLUDE_DIRS}"
    )
endif()
# 🪟🪟🪟🪟








# 🍅🍅🍅🍅 Add .cu files

# # Core CUDA library (no glob here can be safer)
# add_library(cuda_core STATIC
#     src/hello.cu
#     src/scale_kernel.cu
# )

# # Auto-glob version (find all .cu files)
# # slightly dangerous if we delete source files without full rebuild
# file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
# )
# add_library(cuda_core STATIC ${CUDA_SOURCES})



# NEW
# # Recursively find all .cu and .cpp files in src/
# #
file(GLOB_RECURSE CUDA_CPP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    # "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings/*.cpp"
)
# Create a static library from all collected sources
add_library(cuda_core STATIC ${CUDA_CPP_SOURCES}) # NOTE "cuda_core"





# 🍅🍅🍅🍅


# 📦📦📦📦

# Tell CMake that the target `cuda_core` should look for headers in the `include/` folder.
# This does NOT "grab all headers" automatically — it just adds that directory to the compiler's
# search path (-I flag). You still need to #include the headers you want in your .cpp files.

target_include_directories(cuda_core PUBLIC
    include
    core
    # ${CMAKE_CURRENT_SOURCE_DIR}/python/bindings # ❓❓❓❓TRYING TO GET INTELISENSE WORKING
)



# Mark the `cuda_core` target as position-independent code (PIC).
# This is required if you want to link it into shared libraries or Python extension modules,
# because those need relocatable machine code. On Linux, this translates to -fPIC.
set_target_properties(cuda_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Descend into the `python/` subdirectory and process its CMakeLists.txt.
# That’s where you define your Python extension module (`cuda_hello_py`).
add_subdirectory(python)

# 📦📦📦📦


# 🚀🚀🚀🚀 Test Executables

## MANUAL PATTERN:
# add_executable(hello_cu_exec tests/hello_test.cpp) # Define the test executable
# target_link_libraries(hello_cu_exec PRIVATE cuda_core) # Link it against your reusable CUDA core library
# set_target_properties(hello_cu_exec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests) # Tell CMake to place the built binary in build/tests/

## Glob Version
# Collect all .cpp files under tests/
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)

foreach(test_src ${TEST_SOURCES})
    # Get the filename without extension, e.g. hello_test.cpp → hello_test
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE cuda_core)
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
endforeach()

# 🚀🚀🚀🚀