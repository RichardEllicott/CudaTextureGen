
$build_dir = ".\build\"
mkdir -Force $build_dir # make a build folder


Write-Host "================================================================"
Write-Host "compile dll_demo.cpp ..."
Write-Host "----------------------------------------------------------------"
# cl /LD .\dll_demo.cpp -o $build_dir
cl /LD .\dll_demo.cpp /Fe:$build_dir\dll_demo.dll
Write-Host "================================================================"


Write-Host "================================================================"
Write-Host "dumpbin..."
Write-Host "----------------------------------------------------------------"
dumpbin /EXPORTS "$build_dir\dll_demo.dll"
Write-Host "================================================================"


Write-Host "================================================================"
Write-Host "compile main.cpp ..."
Write-Host "----------------------------------------------------------------"
# cl main.cpp -o $build_dir
cl main.cpp /Fe:$build_dir\main.exe
Write-Host "================================================================"


Write-Host "================================================================"
Write-Host "run main.exe ..."
Write-Host "----------------------------------------------------------------"
& "$build_dir\main.exe"
Write-Host "================================================================"
