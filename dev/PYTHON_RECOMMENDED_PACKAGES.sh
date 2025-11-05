
#!/bin/bash
# Script for Ubuntu/Debian systems
# Requires apt, systemd, and bash

# prevent script on non Ubuntu|Debian
if ! grep -q -E 'Ubuntu|Debian' /etc/os-release; then
  echo "This script is intended for Ubuntu or Debian only."
  exit 1
fi

sudo apt update

sudo apt install python3-numpy
sudo apt install python3-pillow
sudo apt install python3-scipy
sudo apt install python3-matplotlib
sudo apt install mypy

# more difficult to see what's installed than with pip
# apt list --installed | grep python3

# 🧹 UPDATE EVERYTHNG! (may as well)
sudo apt update         # Refresh package lists from repositories
sudo apt full-upgrade   # Upgrade all packages, resolving dependencies and replacing obsolete ones
sudo apt install -f     # Fix broken dependencies (if any)
sudo apt autoremove     # Remove packages no longer needed
sudo apt clean          # Clear cached .deb files to free up space


