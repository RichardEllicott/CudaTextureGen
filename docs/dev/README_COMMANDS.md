# Commands

## Windows Powershell

### Misc

```powershell
# remove a file or folder
rm -Recurse -Force "C:\Temp\OldFolder"
```

```powershell
# move to recycle bin
scoop install nircmd # get nircmd with scoop (easier than a native command)
nircmd moverecyclebin .\test\
```

```powershell
# open powershell profile in vs.code
code $PROFILE
```

```powershell
## chocolatey:
## https://chocolatey.org/install
choco list # list installed
choco upgrade all -y # upgrade all (must run in admin shell)
```

```powershell
# adding paths to enviroment (in this case so clang installed by winget can be installed)
$env:PATH = "C:\Program Files\LLVM\bin;" + $env:PATH # add clang to the env so we can find it
$env:PATH = "C:\Program Files\LLVM\lib;" + $env:PATH # and the lib

$env:Path -split ';' # list all the paths (on different lines)
[Environment]::GetEnvironmentVariable('Path','Machine') -split ';' # Machine PATH entries
[Environment]::GetEnvironmentVariable('Path','User') -split ';' # User PATH entries 
```

### scoop

```powershell
# update everything
scoop update
scoop install llvm # install clang
```

### winget

```powershell
winget install --id=LLVM.LLVM -e
```

### WSL

```powershell
# Show distros available to install
wsl --list --online
```

```powershell
# Install Ubuntu 24.04
wsl --install -d Ubuntu-24.04
```

```powershell
# List installed distros with version
wsl -l -v
```

```powershell
# ⚠️ Remove a distro (permanent)
wsl --unregister Ubuntu
```

```powershell
# get Nvidia driver information
nvidia-smi
```

## Linux (Ubuntu-24.04 bash)

### Compress

```bash
# Create a compressed archive
tar -czf archive_name.tar.gz /path/to/directory-or-file
```

```bash
# Decompress archive
tar xzf archive_name.tar.gz
```

### Permissions

```bash
# Ensure the folder and contents are owned by user
sudo chown -R richard:richard /path/to/project 
```

```bash
# Give read/write/execute on directories
find /path/to/project -type d -exec chmod 755 {} \; 
```

```bash
# Give read/write on files
find /path/to/project -type f -exec chmod 644 {} \; 
```

### Misc

```bash
ls - a # list all hidden
```

```bash
# ⚠️ Permanently remove folder and contents
rm -rf folder/
```

```bash
# Get distro version
lsb_release -a
```

```bash
# Update package list
sudo apt update

# Make 'python' run python3 (Ubuntu way)
sudo apt install python-is-python3
```

### Update

```bash
# Update package list
sudo apt update 
# Full update
sudo apt full-upgrade 
# Remove unused packages
sudo apt autoremove
# Clears out the local repository of retrieved package files
sudo apt clean
```

```bash
# apt update and fully upgrade all stuff
sudo apt update; sudo apt full-upgrade; sudo apt autoremove; sudo apt clean
```

```bash
# get scipy
sudo apt update
sudo apt install python3-scipy
```