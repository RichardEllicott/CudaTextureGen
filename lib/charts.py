"""


"""
# import cuda_texture_gen
# import tools


import numpy as np
import matplotlib.pyplot as plt

# Manning-like flux function


def flux(h, S, n=0.05):
    return (1.0 / n) * (h ** (2.0/3.0)) * np.sqrt(S)


# Ranges
depths = np.linspace(0, 2.0, 200)   # water depth [0–2]
slopes = np.linspace(0, 1.0, 200)   # slope [0–1]

# Fixed slopes for depth curves
fixed_slopes = [0.01, 0.05, 0.1, 0.5, 1.0]
# Fixed depths for slope curves
fixed_depths = [0.1, 0.5, 1.0, 2.0]

plt.figure(figsize=(12, 5))

# Flux vs depth
plt.subplot(1, 2, 1)
for S in fixed_slopes:
    plt.plot(depths, flux(depths, S), label=f"Slope={S}")
plt.xlabel("Water depth h")
plt.ylabel("Flux q")
plt.title("Flux vs Water Depth")
plt.legend()

# Flux vs slope
plt.subplot(1, 2, 2)
for h in fixed_depths:
    plt.plot(slopes, flux(h, slopes), label=f"Depth={h}")
plt.xlabel("Slope S")
plt.ylabel("Flux q")
plt.title("Flux vs Slope")
plt.legend()

plt.tight_layout()
plt.show()
