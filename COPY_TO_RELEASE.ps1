# copy the final binary to the library folder
$src = ".\build\windows\src\python\cuda_texture_gen.cp313-win_amd64.pyd"
$dst = ".\lib\cuda_texture_gen\cuda_texture_gen.cp313-win_amd64.pyd"
Copy-Item $src -Destination $dst -Force

# generate a stub to allow code editors to inspect the library
cd .\lib\cuda_texture_gen
stubgen -m cuda_texture_gen --include-private -o . # ends up here
cd ..\..



# ⚠️ this just doesn't want to work!!
# # Copy the built extension into lib/
# $src = ".\build\windows\src\python\cuda_texture_gen.cp313-win_amd64.pyd"
# $dst = ".\lib\cuda_texture_gen\cuda_texture_gen.cp313-win_amd64.pyd"
# Copy-Item $src -Destination $dst -Force

# # Ensure Python can import from lib/
# $env:PYTHONPATH = ".\lib"

# # Generate stubs directly into the package folder
# stubgen -m cuda_texture_gen --include-private -o .\lib\cuda_texture_gen
