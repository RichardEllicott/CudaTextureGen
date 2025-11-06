#!/usr/bin/env python3
"""
platform agnostic build launcher
"""
print("🧙‍♂️ Launching Build...")

import os
import platform
import subprocess
import sys

def main():
    system = platform.system()

    if system == "Windows":
        ps1_script = os.path.join(os.path.dirname(__file__), "BUILD.ps1")
        try:
            # Use PowerShell to run the script
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", ps1_script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ PowerShell script failed: {e}")
            sys.exit(1)

    elif system in ("Linux", "Darwin"):  # Darwin = macOS
        sh_script = os.path.join(os.path.dirname(__file__), "BUILD.sh")
        try:
            subprocess.run(["sh", sh_script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Shell script failed: {e}")
            sys.exit(1)

    else:
        print(f"⚠️ Unsupported OS: {system}")
        sys.exit(1)

    print("🎉 Build complete")

if __name__ == "__main__":
    main()
