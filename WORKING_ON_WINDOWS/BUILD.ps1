






Remove-Item -Recurse -Force .\build
mkdir -Force build # make a build folder





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


$env:INCLUDE += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include"


# cmake -S . -B build -G "Ninja" -DCMAKE_TOOLCHAIN_FILE=./toolchain-msvc.cmake

$toolchain = Resolve-Path ./toolchain-msvc.cmake
cmake -S . -B build -G "Ninja" -DCMAKE_TOOLCHAIN_FILE="$toolchain"
cd build
ninja
# ninja ./build




