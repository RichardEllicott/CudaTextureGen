
# Linux Build Instructions (Ubuntu 24.04 via WSL)

## tested full install 05/11/2025

```powershell
# install linux
wsl --install -d Ubuntu-24.04 
```

```sh
# navigate to repo and run script to install build tools
sudo sh GET_TOOLS.sh
```

```sh
# run build script
sh BUILD.sh
```

```sh
# run test script to confirm build
sh TEST.sh
```

```sh
# optionally run this to copy the build to the ./lib/ and generate a stub
sh COPY_TO_RELEASE.sh
```

## Setup VS.code to work with WSL (optional)

install the "WSL" extension in VS.code, this is if you want to have the repo stored inside linux and work from linux (i tend to work from windows and test on linux)

```bash
sudo apt update
sudo apt install clang-format
```
