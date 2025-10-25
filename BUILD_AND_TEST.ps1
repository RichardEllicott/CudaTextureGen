# build and test

$ErrorActionPreference = "Stop" # like "set -e" in bash (will exit if we crash)

.\BUILD.ps1
.\TEST.ps1