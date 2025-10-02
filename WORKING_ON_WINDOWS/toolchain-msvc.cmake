


# set(CMAKE_RC_COMPILER "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe" CACHE FILEPATH "")
# set(CMAKE_MT "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/mt.exe" CACHE FILEPATH "")
# set(CMAKE_LINKER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/link.exe" CACHE FILEPATH "")




# CUDA compiler
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" CACHE FILEPATH "")

# CUDA host compiler (MSVC cl.exe)
set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" CACHE FILEPATH "")

# C++ compiler
set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" CACHE FILEPATH "")

# Linker
set(CMAKE_LINKER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/link.exe" CACHE FILEPATH "")

# Resource compiler
set(CMAKE_RC_COMPILER "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe" CACHE FILEPATH "")

# Manifest tool
set(CMAKE_MT "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/mt.exe" CACHE FILEPATH "")

# Optional: Windows SDK include/lib paths
set(CMAKE_INCLUDE_PATH "C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/ucrt;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared" CACHE STRING "")
set(CMAKE_LIBRARY_PATH "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x64" CACHE STRING "")




# Windows SDK version — adjust if needed
set(WIN10_SDK_VER "10.0.26100.0")

# Include paths
set(CMAKE_INCLUDE_PATH
    "C:/Program Files (x86)/Windows Kits/10/Include/${WIN10_SDK_VER}/ucrt"
    "C:/Program Files (x86)/Windows Kits/10/Include/${WIN10_SDK_VER}/shared"
    "C:/Program Files (x86)/Windows Kits/10/Include/${WIN10_SDK_VER}/um"
    "C:/Program Files (x86)/Windows Kits/10/Include/${WIN10_SDK_VER}/winrt"
    CACHE STRING "")

# Library paths
set(CMAKE_LIBRARY_PATH
    "C:/Program Files (x86)/Windows Kits/10/Lib/${WIN10_SDK_VER}/ucrt/x64"
    "C:/Program Files (x86)/Windows Kits/10/Lib/${WIN10_SDK_VER}/um/x64"
    CACHE STRING "")




# MSVC runtime libraries (debug and release)
set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH};C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/lib/x64" CACHE STRING "")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /I\"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/include\"" CACHE STRING "")