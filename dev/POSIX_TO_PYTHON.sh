#!/bin/sh
# POSIX-compliant bootstrap

# Check for Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found in PATH"
    exit 1
fi

# Pass all arguments through to Python
exec python3 "$(dirname "$0")/main.py" "$@"


