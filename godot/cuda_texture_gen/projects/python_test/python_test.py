import sys

print("PYTHON READY")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    print(f"ECHO: {line}")
    sys.stdout.flush()
