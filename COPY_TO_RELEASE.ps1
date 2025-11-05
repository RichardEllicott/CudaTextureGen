

# $src = ".\build\windows\src\python\cuda_texture_gen.cp313-win_amd64.pyd"
# $dst = ".\lib\cuda_texture_gen\cuda_texture_gen.cp313-win_amd64.pyd"

# Copy-Item $src -Destination $dst -Force



# stubgen -m cuda_texture_gen --include-private -o ./cuda_texture_gen


# Copy the built extension into lib/
$src = ".\build\windows\src\python\cuda_texture_gen.cp313-win_amd64.pyd"
$dst = ".\lib\cuda_texture_gen\cuda_texture_gen.cp313-win_amd64.pyd"
Copy-Item $src -Destination $dst -Force

# Make sure Python can see lib/
$env:PYTHONPATH = ".\lib"

# Generate stubs into a stubs/ folder at root
stubgen -m cuda_texture_gen --include-private -o .\stubs