# Windows Build Script
#
# REQUIREMENTS:
#
# Visual Studio Installer:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
#
# from the installer, get:
#   - Desktop development with C++
#   - tick the Windows 11 SDK also
#
# should be ~12.35 GB
#
# load the dev shell from the start menu:
#   - "Developer PowerShell for VS 2022"
#
#
# CUDA Toolkit, must be downloaded from Nvidia, currently working on CUDA 12.5 as this is the minimum version working with the 
#   - https://developer.nvidia.com/cuda-12-5-0-download-archive
#
# i use scoop to install my tools, get scoop:
#   - https://scoop.sh/
#
#   - scoop install cmake
#   - scoop install ninja
#   
# Python would need to be installed as admin, this could still be a problem with scoop
#   - scoop uninstall python # if python was not installed global
#   - scoop install python --global
#
#  so i removed it from scoop and use the python website:
#   - https://www.python.org/downloads/ 
#   MAKE SURE TO ADD PYTHON TO PATH!
#
#

# ⚠️ WARNING ⚠️
# getting windows to compile took a lot of added paths to find the right compilers

$ErrorActionPreference = "Stop" # like "set -e" in bash (will exit if we crash)


# ❓❓❓ MAY HAVE BEEN REQUIRED WITHOUT DEV SHELL
# $env:WindowsSdkDir = "C:\Program Files (x86)\Windows Kits\10"
# $env:INCLUDE = @(
#     "$env:WindowsSdkDir\Include\10.0.26100.0\ucrt",
#     "$env:WindowsSdkDir\Include\10.0.26100.0\shared",
#     "$env:WindowsSdkDir\Include\10.0.26100.0\um",
#     "$env:WindowsSdkDir\Include\10.0.26100.0\winrt"
# ) -join ';'


# ❓❓❓ MAY HAVE BEEN REQUIRED WITHOUT DEV SHELL
# $env:LIB = @(
#     "$env:WindowsSdkDir\Lib\10.0.26100.0\ucrt\x64",
#     "$env:WindowsSdkDir\Lib\10.0.26100.0\um\x64"
# ) -join ';'


# ❓❓❓ MAY HAVE BEEN REQUIRED WITHOUT DEV SHELL
# $msvcLibPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64"
# $env:LIB = "$env:LIB;$msvcLibPath"




# ❓❓❓ MAY HAVE BEEN REQUIRED WITHOUT DEV SHELL
# $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
# $env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;" + $env:PATH
# $env:INCLUDE = "$env:CUDA_PATH\include;" + $env:INCLUDE
# $env:LIB = "$env:CUDA_PATH\lib\x64;" + $env:LIB
# $env:CUDACXX = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe"


$build_dir = "build/windows"

# Remove-Item -Recurse -Force .\$buildDir # ⚠️ for debug (force a full rebuild)
mkdir -Force $build_dir # make a build folder

# custom windows toolchain to make work on windows
$toolchain = Resolve-Path ./toolchain-msvc.cmake




# attempting "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" ... note this didn't help my intelisense
# i needed to run windows code from dev console, this also would likely not support CUDA?
# cmake -S . -B build -G "Ninja" -DCMAKE_TOOLCHAIN_FILE="$toolchain" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON


# cmake -S . -B $build_dir -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake -S . -B $build_dir -G "Ninja" -DCMAKE_TOOLCHAIN_FILE="$toolchain" -DCMAKE_BUILD_TYPE=Release


ninja -C $build_dir



