## Command Notes

### WSL

```powershell
# Show distros available to install
wsl --list --online
```
```powershell
# Install Ubuntu 24.04
wsl --install -d Ubuntu-24.04
```
```powershell
# List installed distros with version
wsl -l -v
```
```powershell
# ⚠️ Remove a distro (permanent)
wsl --unregister Ubuntu
```

### Compress
```bash
# Create a compressed archive
tar -czf archive_name.tar.gz /path/to/directory-or-file
```
```bash
# Decompress archive
tar xzf archive_name.tar.gz
```


### Permissions
```bash
# Ensure the folder and contents are owned by user
sudo chown -R richard:richard /path/to/project 
```
```bash
# Give read/write/execute on directories
find /path/to/project -type d -exec chmod 755 {} \; 
```
```bash
# Give read/write on files
find /path/to/project -type f -exec chmod 644 {} \; 
```

### Misc 
```bash
ls - a # list all hidden
```
```bash
# ⚠️ Permanently remove folder and contents
rm -rf folder/
```
```bash
# Get distro version
lsb_release -a
```

```bash
# Update package list
sudo apt update

# Make 'python' run python3 (Ubuntu way)
sudo apt install python-is-python3
```






