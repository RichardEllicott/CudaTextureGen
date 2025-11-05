#!/bin/sh

# get all tools for linux script, designed for Ubuntu, should work on Debian based

# build tools
sudo apt update
sudo apt install -y build-essential cmake ninja-build
sudo apt install -y python3 python3-venv python3-dev
sudo apt install -y nvidia-cuda-toolkit # tested with 12.0.140~12.0.1-4build4


# python tools
sudo apt install python3-numpy
sudo apt install python3-pillow
sudo apt install python3-scipy
sudo apt install python3-matplotlib
sudo apt install mypy


# 🧹 UPDATE EVERYTHNG! (may as well)
sudo apt update         # Refresh package lists from repositories
sudo apt full-upgrade   # Upgrade all packages, resolving dependencies and replacing obsolete ones
sudo apt install -f     # Fix broken dependencies (if any)
sudo apt autoremove     # Remove packages no longer needed
sudo apt clean          # Clear cached .deb files to free up space