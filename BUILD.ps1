param(
    [string]$build_type = "Release",
    [string]$build_dir = "build/windows"
)
$ErrorActionPreference = "Stop" # exit script if any commands crash

# ================================================================================================================================
#region [📝 Docs]
# --------------------------------------------------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------------------------------------------------------------------
#endregion
# ================================================================================================================================

# ================================================================================================================================
#region [✅ Message Symbol]
# --------------------------------------------------------------------------------------------------------------------------------
$IsPS7 = $PSVersionTable.PSVersion.Major -ge 7 # Detect if we're running PowerShell 7+
$msg_symbol = "[!]" # default message symbol is not unicode (to support PowerShell 5.1)

#endregion
# ================================================================================================================================

# ================================================================================================================================
#region [✅ Checks]
# --------------------------------------------------------------------------------------------------------------------------------
if ($IsPS7) { $msg_symbol = [char]0x2705 } # ✅
Write-Output "$msg_symbol Run checks..."

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "Required command 'cmake' not found in PATH. Please install it before continuing."
}
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    throw "Required command 'ninja' not found in PATH. Please install it before continuing."
}
if (Get-Command sccache -ErrorAction SilentlyContinue) { $sccache_found = $true }
#endregion
# ================================================================================================================================

# ================================================================================================================================
#region [📁 Create Build Folder]
# --------------------------------------------------------------------------------------------------------------------------------
if ($IsPS7) { $msg_symbol = [System.Char]::ConvertFromUtf32(0x1F4C1) } # 📁
Write-Output "$msg_symbol Make folder: '$build_dir'"

mkdir -Force $build_dir # make a build folder
# --------------------------------------------------------------------------------------------------------------------------------
#endregion
# ================================================================================================================================

# ================================================================================================================================
#region [🔍 Find Cuda Compiler]
# --------------------------------------------------------------------------------------------------------------------------------
if ($IsPS7) { $msg_symbol = [System.Char]::ConvertFromUtf32(0x1F50D) } # 🔍
Write-Output "$msg_symbol Find Cuda Compiler..."

if (-not $env:CUDA_VARS_LOADED) {

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

    $env:CUDA_VARS_LOADED = "1"
}

# ================================================================================================================================
# ❓ OPTIONAL generate a compile_flags.txt... for clangd to help it find cuda
# After setting $env:CUDA_PATH
# --------------------------------------------------------------------------------------------------------------------------------
if ($IsPS7) { $msg_symbol = [System.Char]::ConvertFromUtf32(0x2753) }  # ❓
Write-Output "$msg_symbol Generate 'compile_flags.txt'..."

$compileFlagsPath = Join-Path $PSScriptRoot "compile_flags.txt"

$flags = @(
    "# Auto-generated by BUILD.ps1 to help clangd find CUDA headers"
    "# Do not edit manually; re-run BUILD.ps1 if CUDA path changes"
    "--cuda-path=$env:CUDA_PATH"
    "-I$env:CUDA_PATH/include"
    "-D__CUDACC__"
    "-nocudalib"
    "-nocudainc"
)

Set-Content -Path $compileFlagsPath -Value $flags -Encoding UTF8
Write-Host "Generated compile_flags.txt for clangd at $compileFlagsPath"

# --------------------------------------------------------------------------------------------------------------------------------
#endregion
# ================================================================================================================================

# ================================================================================================================================
#region [⚙️ Setup Environment]
# --------------------------------------------------------------------------------------------------------------------------------
if ($IsPS7) { $msg_symbol = [System.Char]::ConvertFromUtf32(0x2699) + [System.Char]::ConvertFromUtf32(0xFE0F) + " " }  # ⚙️
Write-Output "$msg_symbol Setup Environment..."

# run VS environment setup script (this is like running "x64 Native Tools Command Prompt for VS 2022")
if (-not $env:VC_VARS_LOADED) { # if we haven't already run this before (speeds up second launch)
    $vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    # Run it with the x64 argument
    cmd /c "`"$vcvars`" x64 && set" | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Item -Force -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
    $env:VC_VARS_LOADED = "1"
}
# --------------------------------------------------------------------------------------------------------------------------------
#endregion
# ================================================================================================================================

# ================================================================================================================================
#region [🔨 Run CMake and Compile]
# --------------------------------------------------------------------------------------------------------------------------------
if ($IsPS7) { $msg_symbol = [System.Char]::ConvertFromUtf32(0x1F528) } # 🔨
Write-Output "$msg_symbol Run CMake and Compile..."

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

# --------------------------------------------------------------------------------------------------------------------------------
#endregion
# ================================================================================================================================
