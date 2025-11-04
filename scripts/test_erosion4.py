"""
testing 

"""
# from pathlib import Path
# import python_bootstrap  # bootstrap to our fresh compiled module
# import cuda_texture_gen
from tools import *

# import inspect
# import os
# from PIL import Image
# import numpy as np
# import math

# from scipy.ndimage import rotate
# from scipy.ndimage import shift


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

print(dir(cuda_texture_gen))


def test_erosion4():

    erosion = cuda_texture_gen.Erosion4()

    folder = "output"
    # folder = "godot/cuda_texture_gen/textures/animation_test"
    base_filename = "test_erosion4"

    array = get_fractal_noise()
    print("min: {}, max: {}".format(array.min(), array.max()))
    normalize_array(array)

    save_array_as_image(array * 255, "{}/{}.png".format(folder, base_filename + ".noise"))

    erosion.steps = 256 * 4
    erosion.mode = 1

    erosion.height_map = array
    erosion.process()
    array = erosion.height_map
    print("min: {}, max: {}".format(array.min(), array.max()))

    normalize_array(array)

    save_array_as_image(array * 255, "{}/{}.png".format(folder, base_filename + ".erode"))

    normal_map = generate_normal_map(array, strength=4.0)
    save_array_as_image(normal_map * 255, "{}/{}.png".format(folder, base_filename + ".erode.normal"))

    ao_map = generate_ao_map(array * 64, radius=3.0, mode=0)
    save_array_as_image(ao_map * 255, "{}/{}.png".format(folder, base_filename + ".erode.ao"))



    albedo = apply_color_map(array)
    save_array_as_image(albedo * 255, "{}/{}.png".format(folder, base_filename + ".albedo"))


    array = erosion.sediment_map
    print("min: {}, max: {}".format(array.min(), array.max()))
    save_array_as_image(array * 255, "{}/{}.png".format(folder, base_filename + ".sediment"))


test_erosion4()
