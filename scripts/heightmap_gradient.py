"""

pip install matplotlib
pip install numpy

"""
from matplotlib import colormaps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tools import *

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np


def test2(width=1024, height=1024):
    function_name = inspect.currentframe().f_code.co_name
    print("⛰️ {}()...".format(function_name))
    filename_base = function_name
    # folder = "output"
    folder = "godot/cuda_texture_gen/textures"

    # noise_generator = cuda_texture_gen.NoiseGenerator()
    # height_map = get_fractal_noise(width, height, 5, 7)
    height_map = get_fractal_noise(width, height, 6, 7)
    normalize_array(height_map)

    height_map = erode_heightmap(height_map)
    normalize_array(height_map)
    blur(height_map)
    normalize_array(height_map)

    save_array_as_image(height_map * 255, "{}/{}{}".format(folder, filename_base, ".png"))

    normal_map = generate_normal_map(height_map, 1.0, True)
    save_array_as_image(normal_map * 255, "{}/{}{}".format(folder, filename_base, ".normal.png"))


    colormap = plt.get_cmap('plasma')  # gist_earth, cividis, viridis, plasma
    colored_map = colormap(height_map)
    save_array_as_image(colored_map * 255, "{}/{}{}".format(folder, filename_base, ".albedo.png"))

    custom_colors = [
        to_rgb("#8fb5f8"),
        to_rgb("#bed27b"),
        to_rgb("#f4a460"),
        to_rgb("#34783a"),
        to_rgb("#adcab1"),
    ]

    rgb_map = smooth_layered_gradient(height_map)
    save_array_as_image(rgb_map * 255, f"{folder}/{filename_base}.layered.smooth.png")

# print(list(colormaps))

test2()


