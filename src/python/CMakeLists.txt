
# python/CMakeLists.txt
#
# bindings can see the whole project but also the nanobind objects 
#

# ================================================================================================================================
# [🐍 Python Bindings]
# --------------------------------------------------------------------------------------------------------------------------------

# Bring in nanobind (either vendored as subdir, or via find_package if installed)
add_subdirectory(nanobind)

# Collect all binding sources in python/bindings/
file(GLOB_RECURSE BINDING_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cpp" # ❓ OPTIONAL
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cu" # ❓ OPTIONAL
)

# Define the Python extension module
# NB: module.cpp is your NB_MODULE entry point
nanobind_add_module(cuda_texture_gen_py
    bindings.cpp # central bind file
    ${BINDING_SOURCES}
)

# Include paths needed for bindings to see core + headers
target_include_directories(cuda_texture_gen_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings
)

# Link against your CUDA static library
target_link_libraries(cuda_texture_gen_py PRIVATE 
    cuda_core
)

# Ensure the output file is named cuda_hello.so / cuda_hello.pyd
set_target_properties(cuda_texture_gen_py PROPERTIES
    OUTPUT_NAME "cuda_texture_gen"
    PREFIX ""  # no "lib" prefix (best for ensuring lib is the same name cross platform)
)