"""




"""
import python_bootstrap  # bootstrap to our fresh compiled module
import cuda_texture_gen
import inspect
import os
from PIL import Image
import numpy as np
from tools import *


os.makedirs("output", exist_ok=True)


def generate_noise_and_erode():

    gen = cuda_texture_gen.NoiseGenerator()

    gen.type = 1
    gen.period = 14
    gen.seed = 0

    print(dir(gen))

    array = gen.generate(1024, 1024)

    normalize_array(array)

    save_array_as_image(array * 255, "output/noise.png")

    print(dir(cuda_texture_gen))

    erosion = cuda_texture_gen.Erosion()
    # erosion = cuda_texture_gen.Erosion2()

    # good settings
    erosion.erosion_rate = 0.01
    erosion.deposition_rate = 0.02 * 0.5
    # lowers squareness if higher as makes sure works on steeper slopes
    erosion.slope_threshold = 0.01
    erosion.steps = 512 * 2
    # erosion.jitter = 0.0



    # erosion.sediment_transport_rate = 1.0 # implicit

    # erosion.jitter = 0.0

    # erosion.min_height = 0.0

    erosion.run_erosion(array)

    save_array_as_image(array * 255, "output/erosion.png")


def erode_array(array):
    erosion = cuda_texture_gen.Erosion2()
    erosion.erosion_rate = 0.01
    erosion.deposition_rate = 0.02 * 0.5
    erosion.slope_threshold = 0.01 # lowers squareness if higher as makes sure works on steeper slopes
    erosion.steps = 512 * 2
    # erosion.jitter = 0.0

    duplicate = np.copy(array)
    erosion.run_erosion(duplicate)
    return duplicate



# using the noise to get fractal nosie (using numpy)
def get_fractal_noise():
    gen = cuda_texture_gen.NoiseGenerator()
    gen.period = 6
    gen.seed = 0
    gen.type = 1

    array = gen.generate(1024, 1024)
    normalize_array(array)

    for i in range(7):
        gen.period += 5
        gen.seed += 1
        array += gen.generate(1024, 1024)

    normalize_array(array)

    return array
    # 

# test macro template design
def test_template_class():
    print(dir(cuda_texture_gen))

    template_class = cuda_texture_gen.TemplateClass()

    print(template_class)
    print(dir(template_class))

    array = get_fractal_noise()
    save_array_as_image(array * 255, "output/fractal_noise.png")

    template_class.test(array)

    save_array_as_image(array * 255, "output/fractal_noise2.png")

# test_template_class()


generate_noise_and_erode()

# array = get_fractal_noise()
# save_array_as_image(array * 255, "output/fractal_noise.png")
# array = erode_array(array)
# save_array_as_image(array * 255, "output/fractal_noise_eroded.png")

