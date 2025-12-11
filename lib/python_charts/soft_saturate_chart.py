"""

illustrating soft saturation 

"""

import numpy as np
import matplotlib.pyplot as plt


def soft_saturate(x, xmax, sharpness):
    return xmax * np.tanh((x / xmax) * sharpness)

# CUDA
"""
float soft_saturate(float x, float xmax, float sharpness = 1.0) {
    return xmax * tanh((x / xmax) * sharpness);
}
"""

sharpness_values = [0.5, 1, 4, 8]


chart_x_range = 3
chart_x_range = 6


# Parameters
xmax = 1.0
x = np.linspace(-chart_x_range * xmax, chart_x_range * xmax, 500)


compute_curves = []
for sharpness in sharpness_values:
    compute_curves.append(soft_saturate(x, xmax, sharpness))


# Plot
plt.figure(figsize=(8, 6))

for i in range(len(sharpness_values)):
    sharpness = sharpness_values[i]
    curve = compute_curves[i]
    plt.plot(x, curve, label=f"sharpness = {sharpness}")




plt.axhline(xmax, color="gray", linestyle="--", linewidth=0.8, label="Ceiling xmax")
plt.axhline(-xmax, color="gray", linestyle="--", linewidth=0.8)

plt.title("Soft Saturation Examples")
plt.xlabel("x")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
