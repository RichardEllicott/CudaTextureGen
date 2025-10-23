# copy_module.ps1

$src = ".\build\windows\python\cuda_texture_gen.cp313-win_amd64.pyd"
$dst = ".\bin\cuda_texture_gen.cp313-win_amd64.pyd"

# Ensure destination folder exists
$dstFolder = Split-Path $dst
if (!(Test-Path $dstFolder)) {
    New-Item -ItemType Directory -Path $dstFolder | Out-Null
}

# Check if destination file exists
if (Test-Path $dst) {
    $response = Read-Host "File already exists at $dst. Overwrite? (y/n)"
    if ($response -ne "y") {
        Write-Host "Copy aborted."
        exit 1
    }
}

# Copy the file
Copy-Item $src -Destination $dst -Force
Write-Host "File copied to $dst."
