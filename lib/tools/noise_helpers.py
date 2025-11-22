"""

standard tools interface, load our library and hook up to it



"""
# import python_bootstrap  # bootstrap to our fresh compiled module
import cuda_texture_gen
import numpy as np
import inspect
# from matplotlib.colors import to_rgb
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap


from .array_helpers import *

def fractal_noise(
        width: int, height: int,
        octaves: int = 6,
        base_period: int = 2,
        seed: int = 12345,
        gain: float = 0.8, lacunarity: float = 2.0):
    """
    get_fractal_noise
    """
    gen = cuda_texture_gen.NoiseGenerator()
    gen.type = 0  # Assuming Perlin or similar

    array = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    period = base_period

    for i in range(octaves):
        gen.period = int(period)
        gen.seed = seed + i

        layer = gen.generate(width, height)

        normalize_array(layer)

        array += layer * amplitude
        total_amplitude += amplitude

        period *= lacunarity
        amplitude *= gain

    array /= total_amplitude  # Normalize final result
    return array


def fractal_noise_rgb(
        width: int, height: int,
        octaves: int = 6,
        base_period: int = 2,
        seed: int = 12345,
        gain: float = 0.8, lacunarity: float = 2.0):

    red = fractal_noise(width, height, octaves, base_period, seed, gain, lacunarity)
    green = fractal_noise(width, height, octaves, base_period, seed + octaves, gain, lacunarity)
    blue = fractal_noise(width, height, octaves, base_period, seed * octaves * 2, gain, lacunarity)

    return np.stack([red, green, blue], axis=-1)
