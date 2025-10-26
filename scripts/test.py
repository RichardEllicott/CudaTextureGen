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

    print("⛰️")

    gen = cuda_texture_gen.NoiseGenerator()

    gen.period = 13
    gen.seed = 0

    print(dir(gen))

    array = gen.generate(1024, 1024)

    normalize_array(array)

    save_array_as_image(array * 255, "output/noise.png")

    print(dir(cuda_texture_gen))

    erosion = cuda_texture_gen.Erosion()
    # erosion = cuda_texture_gen.Erosion2()

    # good settings
    # erosion.erosion_rate = 0.01
    # erosion.deposition_rate = 0.02 * 0.5
    # lowers squareness if higher as makes sure works on steeper slopes
    # erosion.slope_threshold = 0.01
    # erosion.steps = 512 * 2
    # erosion.jitter = 0.0

    # WORKING DEFAULTS
    erosion.mode = 0
    erosion.flow_factor = 0.2
    erosion.erosion_rate = 0.01
    erosion.deposition_rate = 0.02 * 0.5
    erosion.slope_threshold = 0.01
    # erosion.jitter = 0.01
    erosion.wrap = True
    erosion.steps = 512 * 2

    # erosion.mode = 1
    # erosion.rain_rate = 0.005
    # erosion.evaporation_rate = 0.01
    # erosion.flow_factor = 0.2
    # erosion.erosion_rate = 0.01
    # erosion.deposition_rate = 0.5
    # erosion.slope_threshold = 0.005
    # # erosion.jitter = 0.01
    # erosion.wrap = True

    # erosion.mode = 1
    # erosion.flow_factor = 0.2 / 100.0
    # erosion.erosion_rate = 0.01
    # erosion.deposition_rate = 0.02 * 0.5
    # erosion.slope_threshold = 0.01
    # # erosion.jitter = 0.01
    # erosion.wrap = True

    # erosion.rain_rate = 0.005
    # erosion.evaporation_rate = 0.01

    # erosion.sediment_transport_rate = 1.0 # implicit

    # erosion.min_height = 0.0

    erosion.run_erosion(array)

    save_array_as_image(array * 255, "output/erosion.png")


    array2 = erosion.height_map

    save_array_as_image(array * 255, "output/erosion2.png")




def erode_array(array):
    erosion = cuda_texture_gen.Erosion2()
    erosion.erosion_rate = 0.01
    erosion.deposition_rate = 0.02 * 0.5
    # lowers squareness if higher as makes sure works on steeper slopes
    erosion.slope_threshold = 0.01
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

## testing new system for copying data in and out
def test_copy_in_out_water_map():
    print("🫠...")

    gen = cuda_texture_gen.NoiseGenerator()
    print(dir(gen))
    array = gen.generate(1024, 1024)
    normalize_array(array)

    save_array_as_image(array * 255, "output/water_save_test.png")

    erosion = cuda_texture_gen.Erosion()
    print(erosion)
    print(dir(erosion))

    # erosion.set_water_map(array)
    erosion.water_map = array

    # array2 = erosion.get_water_map()
    array2 = erosion.water_map

    print(array2)
    save_array_as_image(array2 * 255, "output/water_save_test2.png")


# test_copy_in_out_water_map()



generate_noise_and_erode() # MAIN TEST ATM

# array = get_fractal_noise()
# save_array_as_image(array * 255, "output/fractal_noise.png")
# array = erode_array(array)
# save_array_as_image(array * 255, "output/fractal_noise_eroded.png")



