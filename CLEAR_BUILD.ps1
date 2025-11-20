# delete the build dir
$ErrorActionPreference = "Stop"

$build_dir = "build/windows"
$clear_build = $true   # set to $false to skip clearing

# Conditionally clear the build directory
if ($clear_build) {
  if (Test-Path $build_dir) {
    Remove-Item -Recurse -Force $build_dir
    Write-Output "Removed $build_dir"
  }
  else {
    Write-Output "No build directory found at $build_dir"
  }
}

