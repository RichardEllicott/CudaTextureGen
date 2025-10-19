# Python C++ CUDA Tests

My C++ CUDA tests with Python bindings

#### Tested using:

- Ubuntu 24.04.3 LTS using WSL
- C++ 17
- CUDA V12.0.140
- Python 3.12.3

#### Also:

- Windows 11
- C++ 17
- CUDA V12.5.40
- Python 3.13.7

## Structure 🚧🚧🚧 (nonsense)

- `src/main.cu`: CUDA kernel and host code
- `BUILD.sh`: Build script using Make
- `CMakeLists.txt`: Simple CMake configuration

## Build Instructions (this changed a bit, WSL is the easiest, Windows use developer shell and pray a bit)

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

to build run

```bash
sh BUILD.sh
```

## Setup VS.code to work with WSL

install the "WSL" extension in VS.code

Linux will need clang-format

```bash
sudo apt update
sudo apt install clang-format
```
