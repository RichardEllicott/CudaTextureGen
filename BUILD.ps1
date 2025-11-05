param(
    [string]$build_type = "Release",
    [string]$build_dir = "build/windows"
)
$ErrorActionPreference = "Stop" # exit script if any commands crash


#region 📝 Docs
# ================================================================================================================================
# 🪟 Windows Build Script
# --------------------------------------------------------------------------------------------------------------------------------
#
# ⚠️ may run from "Developer PowerShell for VS 2022" (but might be okay in normal powershell now)
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
# ================================================================================================================================
#endregion


#region ✅ Checks
# ================================================================================================================================
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "Required command 'cmake' not found in PATH. Please install it before continuing."
}
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    throw "Required command 'ninja' not found in PATH. Please install it before continuing."
}
if (Get-Command sccache -ErrorAction SilentlyContinue) { $sccache_found = $true }
# ================================================================================================================================
#endregion


#region 📁 Create Build Folder

mkdir -Force $build_dir # make a build folder

#endregion


#region 🔍 Find Cuda Compiler
# ================================================================================================================================

# manually point to the CUDA compiler, if not found by the VS environment automaticly
if (-not $env:CUDA_PATH) {
    # fallback if not defined but normally a $env:CUDA_PATH should exist
    $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
    Write-Warning "CUDA_PATH was not defined. Fallback applied: $env:CUDA_PATH"
}

# set up enviroment vars
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;" + $env:PATH
$env:CUDACXX = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
$env:INCLUDE = "$env:CUDA_PATH\include;" + $env:INCLUDE
$env:LIB = "$env:CUDA_PATH\lib\x64;" + $env:LIB

# ================================================================================================================================

#endregion


#region ⚙️ Setup Environment
# ================================================================================================================================
# run VS environment setup script (this is like running "x64 Native Tools Command Prompt for VS 2022")
# --------------------------------------------------------------------------------------------------------------------------------

# Path to the VS environment setup script
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
# Run it with the x64 argument
cmd /c "`"$vcvars`" x64 && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Item -Force -Path "env:$($matches[1])" -Value $matches[2]
    }
}

# ⚠️ optional, if the above runs each time we build, it will create duplicates eventually causing an error
$env:PATH = ($env:PATH -split ';' | Select-Object -Unique) -join ';'
$env:INCLUDE = ($env:INCLUDE -split ';' | Select-Object -Unique) -join ';'
$env:LIB = ($env:LIB -split ';' | Select-Object -Unique) -join ';'
#endregion


#region 🔨 Run CMake and Compile
# ================================================================================================================================

# cmake base starting arguments
$cmake_args = @(
    "-S", ".", "-B", $build_dir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$build_type"
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

# compile with ninja
ninja -C $build_dir

# ================================================================================================================================
#endregion