
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
# CUDA Toolkit, must be downloaded from Nvidia:
#   - https://developer.nvidia.com/cuda-toolkit # version 13 didn't work for me
#   - https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Windows # using 12 for now to match linux
#   - https://developer.nvidia.com/cuda-12-5-0-download-archive?target_os=Windows # 12 didn't work with the MS compiler, 12.5 would be oldest compatible!
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


$env:WindowsSdkDir = "C:\Program Files (x86)\Windows Kits\10"
$env:INCLUDE = @(
    "$env:WindowsSdkDir\Include\10.0.26100.0\ucrt",
    "$env:WindowsSdkDir\Include\10.0.26100.0\shared",
    "$env:WindowsSdkDir\Include\10.0.26100.0\um",
    "$env:WindowsSdkDir\Include\10.0.26100.0\winrt"
) -join ';'

$env:LIB = @(
    "$env:WindowsSdkDir\Lib\10.0.26100.0\ucrt\x64",
    "$env:WindowsSdkDir\Lib\10.0.26100.0\um\x64"
) -join ';'



$msvcLibPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64"
$env:LIB = "$env:LIB;$msvcLibPath"


# $env:INCLUDE += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include"


$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;" + $env:PATH
$env:INCLUDE = "$env:CUDA_PATH\include;" + $env:INCLUDE
$env:LIB = "$env:CUDA_PATH\lib\x64;" + $env:LIB
$env:CUDACXX = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe"





# Remove-Item -Recurse -Force .\build # ⚠️ for debug (force a full rebuild)
mkdir -Force build # make a build folder

$toolchain = Resolve-Path ./toolchain-msvc.cmake
cmake -S . -B build -G "Ninja" -DCMAKE_TOOLCHAIN_FILE="$toolchain"
cd build
ninja




