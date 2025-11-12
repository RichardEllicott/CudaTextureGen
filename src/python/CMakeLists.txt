



#❓❓❓❓ ONLY REQUIRED FOR INTELISENSE??? ❓❓❓❓
# find_package(CUDAToolkit REQUIRED)
# add_library(cuda_hello_py MODULE bindings.cpp)
# target_link_libraries(cuda_hello_py PRIVATE CUDA::cudart cuda_core)
#❓❓❓❓ ONLY REQUIRED FOR INTELISENSE??? ❓❓❓❓
# target_include_directories(python_bindings PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
#❓❓❓❓ ONLY REQUIRED FOR INTELISENSE??? ❓❓❓❓
# find_package(CUDAToolkit)



# 🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️
# SOME NOTES JUST FOR LATER REFERNCE WE CAN ADD VARS, AND SET MESSAGES
# 🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️
# set(MY_VAR "hello")
# message(STATUS "MY_VAR = ${MY_VAR}")
# 🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️🗒️





# --- python/CMakeLists.txt ---

# Bring in nanobind (either vendored as subdir, or via find_package if installed)
add_subdirectory(nanobind)

# Collect all binding sources in python/bindings/
file(GLOB BINDING_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cpp"
)

# Define the Python extension module
# NB: module.cpp is your NB_MODULE entry point
nanobind_add_module(cuda_texture_gen_py
    bindings.cpp
    ${BINDING_SOURCES}
)

# Include paths needed for bindings to see core + headers
target_include_directories(cuda_texture_gen_py PRIVATE
    # ${PROJECT_SOURCE_DIR}/include # ❓❓❓❓
    # ${PROJECT_SOURCE_DIR}/core # ❓❓❓❓
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings
    # ${CMAKE_CURRENT_SOURCE_DIR}/bindings/template_x
)

# Link against your CUDA static library
target_link_libraries(cuda_texture_gen_py PRIVATE 
    cuda_core
    # CUDA::cudart # ❓❓❓❓
)

# Ensure the output file is named cuda_hello.so / cuda_hello.pyd
set_target_properties(cuda_texture_gen_py PROPERTIES
    OUTPUT_NAME "cuda_texture_gen"
    PREFIX ""  # no "lib" prefix (best for ensuring lib is the same name cross platform)
)






# # 🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒
# # THIS VERSION WAS THE OLD ONE, HAD NO SEPERATE /python/bindings folder
# # 🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒

# # Include the nanobind CMake logic (usually from a submodule or local folder)
# # This makes the `nanobind_add_module()` function available
# add_subdirectory(nanobind)

# # Define a Python extension module named `cuda_hello_py`
# # This compiles `bindings.cpp` and wraps it as a `.pyd` (Windows) or `.so` (Linux/macOS)
# nanobind_add_module(cuda_hello_py bindings.cpp)

# # Link the Python module against your CUDA static library
# # This allows Python to call into CUDA kernels or host-side logic defined in `cuda_core`
# target_link_libraries(cuda_hello_py PRIVATE cuda_core)

# # Rename the output file to `cuda_hello` so you can `import cuda_hello` in Python
# set_target_properties(cuda_hello_py PROPERTIES OUTPUT_NAME "cuda_hello")

# # 🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒

