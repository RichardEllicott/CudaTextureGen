# CUDA Hello World

This is a minimal CUDA + C++ project that demonstrates a simple GPU kernel launch.
Written on WSL using Ubuntu 24.04.3 LTS

## Structure

- `src/main.cu`: CUDA kernel and host code
- `BUILD.sh`: Build script using Make
- `BUILD_NINJA.sh`: Build script using Ninja
- `CMakeLists.txt`: Simple CMake configuration

## Build Instructions

```bash
./BUILD.sh
```


## Commands

wsl --list --online # list linux distros that can be installed
wsl --install -d Ubuntu-24.04 # install Ubuntu-24.04


wsl -l -v # list wsl linux's installed
wsl --unregister Ubuntu # delete installed linux


tar -czvf archive-name.tar.gz /path/to/directory-or-file # compress (verbose)
tar -czf archive-name.tar.gz /path/to/directory-or-file # compress
tar xzf file.tar.gz # decompress

ls - a # list all hidden
rm -rf folder/ # remove folder 




## Install Notes

wsl --install -d Ubuntu-24.04 # install Ubuntu 24.04 (mainly as we need python 3.12)


sudo apt update
sudo apt install -y build-essential cmake ninja-build
sudo apt install -y python3 python3-venv python3-dev
sudo apt install -y nvidia-cuda-toolkit





```bash
pip install -e . --break-system-packages
```



