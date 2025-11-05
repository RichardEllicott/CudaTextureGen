# pip upgrade all python packages shortcut (windows)

pip list --outdated | ForEach-Object { $_.Split()[0] } | ForEach-Object { pip install --upgrade $_ }