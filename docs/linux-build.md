
## Linux Build Instructions (Ubuntu 24.04 via WSL)


install Ubuntu 24.04

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

### Setup VS.code to work with WSL

install the "WSL" extension in VS.code

Linux will need clang-format

```bash
sudo apt update
sudo apt install clang-format
```
