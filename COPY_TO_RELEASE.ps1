
# $ErrorActionPreference = "Stop" # exit on crash


# # copy the final binary to the library folder
# $src = ".\build\windows\src\python\cuda_texture_gen.cp313-win_amd64.pyd"
# $dst = ".\lib\cuda_texture_gen\cuda_texture_gen.cp313-win_amd64.pyd"
# Copy-Item $src -Destination $dst -Force

# # # generate a stub to allow code editors to inspect the library
# # cd .\lib\cuda_texture_gen
# # stubgen -m cuda_texture_gen --include-private -o . # ends up here
# # cd ..\..

# # should be more reliable if we crash in that it will return to current directory
# $old_dir = Get-Location
# # move into the package folder, run stubgen, then return
# Set-Location .\lib\cuda_texture_gen
# stubgen -m cuda_texture_gen --include-private -o .
# Set-Location $old_dir



# always resolve paths relative to the script file itself
$src = Join-Path $PSScriptRoot "build\windows\src\python\cuda_texture_gen.cp313-win_amd64.pyd"
$dst = Join-Path $PSScriptRoot "lib\cuda_texture_gen\cuda_texture_gen.cp313-win_amd64.pyd"
Copy-Item $src -Destination $dst -Force

# save current directory
$old_dir = Get-Location

try {
    # move into the package folder, run stubgen, then return
    Set-Location (Join-Path $PSScriptRoot "lib\cuda_texture_gen")
    stubgen -m cuda_texture_gen --include-private -o .
}
finally {
    Set-Location $old_dir
}





