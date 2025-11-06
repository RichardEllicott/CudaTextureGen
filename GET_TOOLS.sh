#!/bin/sh
# ================================================================================================================================
# 💡 Setup Linux Tools, tested with Ubuntu-24.04
# ================================================================================================================================

#region 🔨 Build Tools
# ================================================================================================================================
apt update -y
apt install -y build-essential cmake ninja-build
apt install -y python3 python3-venv python3-dev
apt install -y nvidia-cuda-toolkit # tested with 12.0.140~12.0.1-4build4
# sudo snap install sccache --classic # get sccache, not yet implemented in linux
# ================================================================================================================================
#endregion

#region 🐍 Python Tools
# ================================================================================================================================
apt install mypy -y # required for stubgen (allowing python code inteligence)
apt install python3-numpy -y # required
apt install python3-imageio -y # used to save images
apt install python3-pil -y # sometimes pillow is used by imageio for png


# apt install python3-scipy -y # optional, useful for manipulating arrays
# apt install python3-matplotlib -y # optional, useful for it's color gradients
# ================================================================================================================================
#endregion

#region 🧹 Update All and Clean Up
# ================================================================================================================================
apt update -y            # Refresh package lists
apt full-upgrade -y      # Upgrade all packages
apt install -f -y        # Fix broken dependencies
apt autoremove -y        # Remove packages no longer needed
apt clean                # Clear cached .deb files
# ================================================================================================================================
#endregion
