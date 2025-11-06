"""




"""
from pathlib import Path
import python_bootstrap  # bootstrap to our fresh compiled module
import cuda_texture_gen
import inspect
import os
from PIL import Image
import numpy as np
from tools import *
import math

from scipy.ndimage import rotate
from scipy.ndimage import shift


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

os.makedirs("output", exist_ok=True)

print(dir(cuda_texture_gen))

# was attemting to rotate value noise (no luck)
def test_noise():
    noise = cuda_texture_gen.Noise()

    print(dir(noise))

    noise.type = 0
    
    noise.process()
    array = noise.image

    print("mind: {}, max: {}".format(array.min(), array.max()))


    offset_array(array)
    normalize_array(array)
    save_array_as_image(array * 255, "output/noise_test1.png")


    # noise.angle = 45.0
    noise.x = 0.5
    noise.y = 0.5
    noise.z = 1.0
    noise.process()
    array = noise.image
    print("mind: {}, max: {}".format(array.min(), array.max()))

    # offset_array(array)
    normalize_array(array)
    save_array_as_image(array * 255, "output/noise_test2.png")


# test_noise()

# testing 3d gradient noise
def test_noise_animation():


    # folder = "output/"
    folder = "godot/cuda_texture_gen/textures/animation_test"
    base_filename = "noise_frame"


    noise = cuda_texture_gen.Noise()

    noise.period = 27

    noise.period_x = 27

    noise.width = 512
    noise.height = 512

    frames = 32
    div = 1.0 / frames

    for i in range(frames):
        print("make frame {}".format(i))

        print("noise.z = {}".format(noise.z))

        noise.process()
        array = noise.image
        array *= 1.5

        array /= 2.0
        array += 0.5

        print("min: {}, max: {}".format(array.min(), array.max()))

        
        save_array_as_image(array * 255, "{}/{}_{:02}.png".format(folder, base_filename, i))

        noise.z += div
        


test_noise_animation()
    