param(
    # [switch]$sccache,
    [string]$config = "Release"
)
#
# 🪟 Windows Build Script
#
# ⚠️ run from "Developer PowerShell for VS 2022"
#
# requirements:
# 
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# from the installer, get:
#   - Desktop development with C++
#   - tick the Windows 11 SDK also
# should be ~12.35 GB
#
# get CUDA Toolkit
# https://developer.nvidia.com/cuda-12-5-0-download-archive
#
#
# other tools can be got from scoop
# https://scoop.sh/
#
# scoop install cmake
# scoop install ninja
# scoop install sccache # optional
#
# python from scoop has issues, best to install from website:
# https://www.python.org/downloads/ (tested on 3.13.7)
#

$ErrorActionPreference = "Stop" # like "set -e" in bash (will exit if we crash)


# ----------------------------------------------------------------
# ensure build directory exists
$build_dir = "build/windows"
mkdir -Force $build_dir # make a build folder
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# ⚙️ we need to manually point to the CUDA compiler, this is not found by the VS environment automaticly
if (-not $env:CUDA_PATH) {
    # fallback if not defined but normally a $env:CUDA_PATH should exist
    $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
}
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;" + $env:PATH
$env:INCLUDE = "$env:CUDA_PATH\include;" + $env:INCLUDE
$env:LIB = "$env:CUDA_PATH\lib\x64;" + $env:LIB
$env:CUDACXX = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# ⚙️ this is like running the "x64 Native Tools Command Prompt for VS 2022" and solves most windows build issues
# the following allows me to run from "Developer PowerShell for VS 2022"

# Path to the VS environment setup script
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
# Run it with the x64 argument
cmd /c "`"$vcvars`" x64 && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Item -Force -Path "env:$($matches[1])" -Value $matches[2]
    }
}

# ⚠️ this part is optional, if the above runs each time we build, it will create duplicates
$env:PATH = ($env:PATH -split ';' | Select-Object -Unique) -join ';'
$env:INCLUDE = ($env:INCLUDE -split ';' | Select-Object -Unique) -join ';'
$env:LIB = ($env:LIB -split ';' | Select-Object -Unique) -join ';'

# ----------------------------------------------------------------


# ----------------------------------------------------------------
if (Get-Command cmake -ErrorAction SilentlyContinue) {
    Write-Host "cmake found!"
    $cmake_found = $true
}
if (Get-Command ninja -ErrorAction SilentlyContinue) {
    Write-Host "ninja found!"
    $ninja_found = $true
}

if (Get-Command sccache -ErrorAction SilentlyContinue) {
    Write-Host "sccache found!"
    $sccache_found = $true
}
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# cmake base starting arguments
$cmake_args = @(
    "-S", ".", "-B", $build_dir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$config"
)

# conditionally add sccache launchers
if ($sccache_found) {
    $cmake_args += @(
        "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
        "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache"
    )
}
else {
    Write-Host "*WARNING* sccache not found, building without it"
}

# run cmake
cmake @cmake_args

# ❓ old command, kept as notes
# cmake -S . -B $build_dir -G "Ninja" `
#     "-DCMAKE_BUILD_TYPE=$config" `
#     -DCMAKE_C_COMPILER_LAUNCHER=sccache `
#     -DCMAKE_CXX_COMPILER_LAUNCHER=sccache `
#     -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache

# 🏗️ finally compile
ninja -C $build_dir
# ----------------------------------------------------------------

