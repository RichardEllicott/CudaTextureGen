# Cuda Texture Generation

CudaTextureGeneration is a Python-based framework for procedural texture generation and erosion simulation, accelerated with CUDA. It provides modular tools for crafting realistic surface patterns, simulating terrain wear, and integrating GPU-powered workflows into creative or scientific pipelines.

## ⚙️ Requirements

This framework requires a CUDA-capable NVIDIA GPU and a compatible
software stack. It has been tested on the following configurations:

### ✅ Primary Environment

* OS: Ubuntu 24.04.3 LTS (via WSL)
* C++: C++17
* CUDA: V12.0.140
* Python: 3.12.3

### 🧪 Also Tested On

* OS: Windows 11
* C++: C++17
* CUDA: V12.5.40
* Python: 3.13.7

> ⚠️ **A CUDA-compatible GPU is required.** This project will not run on systems without NVIDIA hardware and drivers.  
> 🧪 *Note: Future support for non-CUDA backends may be possible via [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp), a SYCL-based compiler targeting multiple accelerators.*

## Structure

This repository is organized as follows:

* `src/`: CUDA kernel and host code  
* `include/`: Header files  
* `core/`: Core header library (currently unused)  
* `python/`: Python bindings directory  
* `python/bindings/`: Individual binding modules  
* `BUILD.sh`: Build script using CMake (Linux)  
* `BUILD.ps1`: Build script using CMake (Windows)  
* `CMakeLists.txt`: Simple CMake configuration
* `test.py`: Some simple tests

## Building

For platform-specific build instructions, see:

* [Windows Build Guide](docs/windows-build.md)
* [Linux Build Guide](docs/linux-build.md)
