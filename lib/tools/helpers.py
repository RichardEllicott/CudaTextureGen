"""

standard tools interface, load our library and hook up to it



"""
# import python_bootstrap  # bootstrap to our fresh compiled module
import cuda_texture_gen
import numpy as np
import inspect
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap


from .array_helpers import *


def print_current_function():
    """
    print function call
    """
    frame = inspect.currentframe().f_back  # caller's frame
    func_name = frame.f_code.co_name
    args, _, _, values = inspect.getargvalues(frame)
    arg_str = ', '.join(f"{arg}={values[arg]!r}" for arg in args)
    print(f"{func_name}({arg_str})...")


# # Split into RGB channels
# R = arr[:, :, 0]  # Red channel
# G = arr[:, :, 1]  # Green channel
# B = arr[:, :, 2]  # Blue channel


def erode_heightmap(height_map,
                    rain_rate=0.002,
                    max_water_outflow=0.005,
                    capacity=0.5,
                    erode=0.02,
                    deposit=0.005,
                    evaporation_rate=0.006,
                    steps=1024,
                    wrap=True

                    ):
    """
    run erosion
    """
    erosion = cuda_texture_gen.Erosion3()

    erosion.rain_rate = rain_rate
    erosion.max_water_outflow = max_water_outflow
    erosion.capacity = capacity
    erosion.erode = erode
    erosion.deposit = deposit
    erosion.evaporation_rate = evaporation_rate
    erosion.steps = steps
    erosion.wrap = wrap

    erosion.height_map = height_map
    erosion.process()
    height_map = erosion.height_map

    return height_map


def generate_normal_map(array, strength=1.0, wrap=True):
    """
    generate normal map
    """
    return cuda_texture_gen.generate_normal_map(array, strength, wrap)


def generate_ao_map(array, radius=1.0, wrap=True, mode=0):
    """
    generate ambient occlusion
    """
    return cuda_texture_gen.generate_ao_map(array, radius, wrap, mode)


def smooth_layered_gradient(height_map, band_colors=[
        to_rgb("#8fb5f8"),
        to_rgb("#bed27b"),
        to_rgb("#f4a460"),
        to_rgb("#34783a"),
        to_rgb("#adcab1"),
    ]
):
    """
    height_map: 2D array normalized to [0, 1]
    band_colors: list of RGB tuples (e.g., from to_rgb)
    """
    bands = np.linspace(0, 1, len(band_colors))
    rgb_map = np.zeros((*height_map.shape, 3), dtype=np.float32)

    for i in range(len(bands) - 1):
        lower, upper = bands[i], bands[i + 1]
        mask = (height_map >= lower) & (height_map <= upper)
        t = (height_map[mask] - lower) / (upper - lower)  # blend factor [0,1]

        c0 = np.array(band_colors[i])
        c1 = np.array(band_colors[i + 1])
        rgb_map[mask] = (1 - t)[:, None] * c0 + t[:, None] * c1

    return rgb_map


# HAS A PROBLEM!!
def blur_array_cuda(input, amount=1.0, wrap=True):
    cuda_texture_gen.blur(input, amount, wrap)



def apply_color_map(height_map, cmap="terrain"):
    """
    apply matplotlib colormap

    Choose a colormap (e.g., 'terrain', 'viridis', 'plasma', or define your own)
    """
    colormap = plt.get_cmap(cmap)
    return colormap(height_map)
