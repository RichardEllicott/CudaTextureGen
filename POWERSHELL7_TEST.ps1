# Requires PowerShell 7+
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Error "This script requires PowerShell 7 or later. Current version: $($PSVersionTable.PSVersion)"
    exit 1
}

Write-Host "Running in PowerShell $($PSVersionTable.PSVersion) [$($PSVersionTable.PSEdition)]"
