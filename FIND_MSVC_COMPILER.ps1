
# script to locate MSVC compiler

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$clPath = Join-Path $vsPath "VC\Tools\MSVC"
$version = Get-ChildItem $clPath | Sort-Object Name -Descending | Select-Object -First 1
$compiler = Join-Path $version.FullName "bin\Hostx64\x64\cl.exe"


Write-Host "MSVC compiler path: $compiler"


$env:CL_PATH = $compiler # write it to a var


# you could then use it with cmake 
# -DCMAKE_CXX_COMPILER="$env:CL_PATH"