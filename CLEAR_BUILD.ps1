# delete the build dir
$ErrorActionPreference = "Stop"

$build_dir = "build/windows"

$clear_build = $true   # set to $false to skip clearing

# Conditionally clear the build directory
if ($clear_build -and (Test-Path $build_dir)) {
  Remove-Item -Recurse -Force $build_dir
}
