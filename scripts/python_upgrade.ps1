

# pip list --outdated --format=freeze > outdated.txt
# pip install --upgrade -r outdated.txt


pip list --outdated | ForEach-Object { $_.Split()[0] } | ForEach-Object { pip install --upgrade $_ }

