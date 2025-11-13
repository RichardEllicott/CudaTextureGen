cmake_minimum_required(VERSION 3.18)
project(cuda_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # will generate compile_commands.json for vs.code

# testing debug 🚧 set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo
# -DDEBUG") # 🚧


# # ⚠️⚠️⚠️⚠️ ATTEMPT TO SUPPORT clangd.. this actually did add the library path to the 
# include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include")
# # ⚠️⚠️⚠️⚠️

# ================================================================================================================================
# 🐍 Find Python
# --------------------------------------------------------------------------------------------------------------------------------

find_package(Python REQUIRED COMPONENTS Interpreter Development)

# 🪟 required for windows
if(WIN32 AND NOT TARGET Python::Module)
  add_library(Python::Module UNKNOWN IMPORTED)
  set_target_properties(
    Python::Module
    PROPERTIES IMPORTED_LOCATION "${Python_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Python_INCLUDE_DIRS}")
endif()

# ================================================================================================================================
# 🍅 Add cu/cpp source files
# --------------------------------------------------------------------------------------------------------------------------------

# Recursively find all .cu and .cpp files in src/main/ note GLOB_RECURSE instead
# of GLOB
file(
  GLOB_RECURSE
  CUDA_CPP_SOURCES
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/*.cpp"
  # "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings/*.cpp"
)
# Create a static library from all collected sources
add_library(cuda_core STATIC ${CUDA_CPP_SOURCES}) # NOTE "cuda_core"

# ================================================================================================================================
# 📦 add include directories and our python bits
# --------------------------------------------------------------------------------------------------------------------------------

# Tell CMake that the target `cuda_core` should look for headers in the
# `include/` folder. This does NOT "grab all headers" automatically — it just
# adds that directory to the compiler's search path (-I flag). You still need to
# #include the headers you want in your .cpp files.

target_include_directories(
  cuda_core
  PUBLIC src/include # headers
         src/core # core helpers, mainly templates to manage cuda
         src/kernels # some extra kernel only files
)

# Mark the `cuda_core` target as position-independent code (PIC). This is
# required if you want to link it into shared libraries or Python extension
# modules, because those need relocatable machine code. On Linux, this
# translates to -fPIC.
set_target_properties(cuda_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Descend into the `python/` subdirectory and process its CMakeLists.txt. That’s
# where you define your Python extension module (`cuda_hello_py`).
add_subdirectory(src/python)

# ================================================================================================================================
# 🚀 Test Executables (don't rely on python at all)
# --------------------------------------------------------------------------------------------------------------------------------

option(ENABLE_TESTS "Build the test executables" ON)

if(ENABLE_TESTS)

  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp) # get each cpp file in tests

  foreach(test_src ${TEST_SOURCES})
    # Get the filename without extension, e.g. hello_test.cpp → hello_test
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE cuda_core)
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  endforeach()

endif()

# ================================================================================================================================
# 🔗 testing cross platform dlls (not using boost or sane they broke)
# --------------------------------------------------------------------------------------------------------------------------------

# note this comes out in the windows folder, top level

option(ENABLE_PLUGIN_TESTS "Build plugin test executables and DLLs" ON)
if(ENABLE_PLUGIN_TESTS)

  # Plugin
  add_library(plugin SHARED src/plugin_test/plugin/plugin.cpp)
  set_target_properties(
    plugin PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugin_test
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugin_test)
  target_link_libraries(plugin PRIVATE cuda_core)

  # Loader
  add_executable(loader src/plugin_test/loader/loader.cpp)
  set_target_properties(loader PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugin_test)
  target_link_libraries(loader PRIVATE cuda_core)

endif()
