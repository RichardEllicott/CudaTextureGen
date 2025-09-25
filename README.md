# Python C++ CUDA Tests

My C++ CUDA tests with Python bindings

Written on Ubuntu 24.04.3 LTS using WSL 
C++17
CUDA 12.0
Python 3.12


## Structure

- `src/main.cu`: CUDA kernel and host code
- `BUILD.sh`: Build script using Make
- `BUILD_NINJA.sh`: Build script using Ninja
- `CMakeLists.txt`: Simple CMake configuration



## Build Instructions

install Ubuntu 24.04 using WSL

```powershell
wsl --install -d Ubuntu-24.04 
```

install build tools

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build
sudo apt install -y python3 python3-venv python3-dev
sudo apt install -y nvidia-cuda-toolkit
```

get python libraries

```bash
sudo apt install python3-pil
sudo apt install python3-numpy
```

to build run the script

```bash
sh BUILD.sh
```

## Commands

```powershell
wsl --list --online # list linux distros that can be installed
wsl --install -d Ubuntu-24.04 # install Ubuntu-24.04
wsl -l -v # list wsl linux's installed
wsl --unregister Ubuntu # delete installed linux
```

```powershell
tar -czvf archive-name.tar.gz /path/to/directory-or-file # compress (verbose)
tar -czf archive-name.tar.gz /path/to/directory-or-file # compress
tar xzf file.tar.gz # decompress
```

```powershell
ls - a # list all hidden
rm -rf folder/ # remove folder 

```

