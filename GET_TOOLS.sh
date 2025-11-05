#!/bin/sh

# get all tools for linux script, designed for Ubuntu, should work on Debian based

# build tools
apt update -y
apt install -y build-essential cmake ninja-build
apt install -y python3 python3-venv python3-dev
apt install -y nvidia-cuda-toolkit # tested with 12.0.140~12.0.1-4build4


# # python tools
# apt install python3-numpy -y # required
# apt install python3-pillow -y # recommended (to save images)

apt install mypy -y # required stubgen

# apt install python3-scipy -y # optional, useful for manipulating arrays
# apt install python3-matplotlib -y # optional, useful for it's color gradients


# 🧹 UPDATE EVERYTHING! (unattended)
apt update -y            # Refresh package lists
apt full-upgrade -y      # Upgrade all packages
apt install -f -y        # Fix broken dependencies
apt autoremove -y        # Remove packages no longer needed
apt clean                # Clear cached .deb files
