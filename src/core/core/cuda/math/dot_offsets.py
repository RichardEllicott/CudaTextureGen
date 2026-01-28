"""



"""


import math

offsets = [
    (1, 0), (-1, 0),
    (0, 1), (0, -1),
    (1, 1), (-1, -1),
    (1, -1), (-1, 1)
]

normals = []

total_length = 0.0

for ox, oy in offsets:
    length = math.sqrt(ox*ox + oy*oy)
    if length != 0:
        nx, ny = ox/length, oy/length
    else:
        nx, ny = 0.0, 0.0

    total_length += length

    print(f"offset=({ox:2d}, {oy:2d})  length={length:.6f}  norm=({nx:.6f}, {ny:.6f})")

    normals.append((nx, ny))

print(f"total_length={total_length:.6f}")




normals_scaled = [(n[0] / total_length, n[1] / total_length)for n in normals]

s = "{"

for x, y in normals_scaled:
    s += f"{{{x}, {y}}}, "

s = s.rstrip(", ")  # remove trailing comma + space
s += "};\n"

print(s)
