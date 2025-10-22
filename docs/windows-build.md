
## 🪟 Windows Build Instructions

Building on Windows is more involved and may depend on your specific environment. The following setup has been tested and confirmed to work:

### 1. Install Visual C++ Build Tools

Download and install from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

From the installer, select:

* ✅ **Desktop development with C++**
* ✅ **Windows 11 SDK**

### 2. Install CUDA Toolkit

Download from [NVIDIA CUDA 12.5 Archive](https://developer.nvidia.com/cuda-12-5-0-download-archive)

* Ensure your **GPU drivers are up to date**
* Confirm CUDA is correctly installed and available in your environment

### 3. Launch Developer Shell

Open the **Developer PowerShell for VS 2022** from the Start Menu. This ensures the correct environment variables are set.

### 4. Install Build Tools via Scoop

Install [Scoop](https://scoop.sh/) — a simple Windows package manager.

Then run:

```powershell
scoop install cmake
scoop install ninja
```

### 5. Install Python

Download from [python.org](https://www.python.org/downloads/)

> ⚠️ Be sure to **add Python to your PATH** during installation.

---

Once all dependencies are installed, you can run the build script:

```powershell
./BUILD.ps1
```

> 🧪 Note: This setup is tailored to the author's Windows environment. Compatibility may vary across systems. To debug see `BUILD.ps1` and `toolchain-msvc.cmake` where various paths have been hardcoded
