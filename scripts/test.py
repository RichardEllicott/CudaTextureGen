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

os.makedirs("output", exist_ok=True)


def get_fractal_noise(width=1024, height=1024, octaves=6, base_period=2, base_seed=0, gain=0.8, lacunarity=2.0):
    gen = cuda_texture_gen.NoiseGenerator()
    gen.type = 0  # Assuming Perlin or similar

    array = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    period = base_period

    for i in range(octaves):
        gen.period = int(period)
        gen.seed = base_seed + i

        layer = gen.generate(width, height)
        normalize_array(layer)

        array += layer * amplitude
        total_amplitude += amplitude

        period *= lacunarity
        amplitude *= gain

    array /= total_amplitude  # Normalize final result
    return array

# gen some gradient noise, and make a normal map and ao map


def test_shader_maps(filename="output/04_eroded.png"):

    shader_maps = cuda_texture_gen.ShaderMaps()
    print(shader_maps)
    print(dir(shader_maps))

    print("load image...")
    img = Image.open(filename).convert("L")
    arr = np.array(img, dtype=np.float32)

    normal_arr = shader_maps.generate_normal_map(arr)
    ao_arr = shader_maps.generate_ao_map(arr * 0.5, radius=2)

    path = Path(filename)
    save_array_as_image(
        normal_arr * 255.0, str(path.with_name(path.stem + ".normal" + path.suffix)))
    save_array_as_image(
        ao_arr * 255.0, str(path.with_name(path.stem + ".ao" + path.suffix)))


def test_resample():

    shader_maps = cuda_texture_gen.ShaderMaps()
    noise_generator = cuda_texture_gen.NoiseGenerator()
    resample = cuda_texture_gen.Resample()
    erosion = cuda_texture_gen.Erosion2()

    # noise = get_fractal_noise(width=1024, height=1024, octaves=6, base_period=2, base_seed=0, gain=0.8, lacunarity=2.0)
    noise = get_fractal_noise(width=1024, height=1024, octaves=3, base_period=7, base_seed=0, gain=0.8, lacunarity=1.9)
    # noise = noise_generator.generate(1024, 1024)

    normalize_array(noise)

    save_array_as_image(noise * 255, "output/00_noise.png")

    # normal_map = shader_maps.generate_normal_map(noise, 1.0, True)
    normal_map = shader_maps.generate_normal_map(noise)
    save_array_as_image(normal_map * 255, "output/00_noise.normal.png")

    noise_generator.period = 5
    noise_generator.seed = 0

    map_x = noise_generator.generate(1024, 1024)
    normalize_array(map_x)
    save_array_as_image(map_x * 255, "output/01_map_x.png")

    noise_generator.seed += 1
    map_y = noise_generator.generate(1024, 1024)
    normalize_array(map_y)
    save_array_as_image(map_y * 255, "output/01_map_y.png")

    #
    #
    #
    # Convert normalized float arrays to uint8
    red_channel = (map_x * 255).astype(np.uint8)
    green_channel = (map_y * 255).astype(np.uint8)
    blue_channel = np.zeros_like(red_channel, dtype=np.uint8)

    # Stack into RGB image
    rgb_array = np.stack((red_channel, green_channel, blue_channel), axis=-1)

    # Save using PIL
    image = Image.fromarray(rgb_array, mode='RGB')
    image.save('output/02_map_xy.png')
    #
    #
    #

    resample.input = noise

    resample.map_x = map_x * 64
    resample.map_y = map_y * 64

    resample.process()

    resampled = resample.output
    save_array_as_image(resampled * 255, "output/03_resampled.png")

    erosion.height_map = resampled
    erosion.steps = 64 * 4
    erosion.process()

    eroded = erosion.height_map
    save_array_as_image(eroded * 255, "output/04_eroded.png")

    print(shader_maps)

    print(type(eroded))

    normal_map = shader_maps.generate_normal_map(eroded, 1.0, True)

    test_shader_maps("output/04_eroded.png")

    test_blur("output/04_eroded.ao.png", "output/04_eroded.ao.blur.png")


# we HAD a bug!!! but don't seem to here!?!?
def test_shader_maps2(width=1024, height=1024):

    filename_base = inspect.currentframe().f_code.co_name

    print("🐢" * 4, "test_shader_maps2()...")
    shader_maps = cuda_texture_gen.ShaderMaps()
    print(shader_maps)
    print(dir(shader_maps))
    print("🐢" * 4)

    noise_generator = cuda_texture_gen.NoiseGenerator()
    noise_generator.period = 27

    array = noise_generator.generate(width, height)
    normalize_array(array)

    save_array_as_image(array * 255, "output/{}.png".format(filename_base))

    normal_map = shader_maps.generate_normal_map(array)
    save_array_as_image(normal_map * 255, "output/{}.normal.png".format(filename_base))

    ao_map = shader_maps.generate_ao_map(array)
    save_array_as_image(ao_map * 255, "output/{}.ao.png".format(filename_base))


# test_resample()

# test_shader_maps2()


# print(dir(cuda_texture_gen))
# template_base_1_test = cuda_texture_gen.TemplateBase1Test()
# print(dir(template_base_1_test))
# template_base_1_test.process()


def test_template_class_3():
    function_name = inspect.currentframe().f_code.co_name
    print("🐮 {}()...".format(function_name))
    filename_base = function_name

    noise_generator = cuda_texture_gen.NoiseGenerator()
    template_class_3 = cuda_texture_gen.TemplateClass3()

    array = noise_generator.generate(1024, 1024)
    normalize_array(array)
    save_array_as_image(array * 255, "output/{}_0.png".format(filename_base))

    # template_class_3.image = array
    # template_class_3.process()
    # array = template_class_3.image

    array = template_class_3.process(array)

    save_array_as_image(array * 255, "output/{}_1.png".format(filename_base))

# test_template_class_3()


def test_blur(filename, output_filename, amount=1):
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    print("load image...")
    img = Image.open(filename).convert("L")
    arr = np.array(img, dtype=np.float32)
    print("height range: [{}, {}]".format(arr.min(), arr.max()))
    print("blur...")

    cuda_texture_gen.blur(arr, amount=1, wrap=True)

    save_array_as_image(arr, output_filename)


def test_erosion_3():
    function_name = inspect.currentframe().f_code.co_name
    print("⛰️ {}()...".format(function_name))
    filename_base = function_name

    array = get_fractal_noise(1024, 1024, 6, 7)
    array = get_fractal_noise(1024, 1024, 2, 7)
    array = get_fractal_noise(1024, 1024, 2, 3)
    normalize_array(array)
    save_array_as_image(array * 255, "output/{}_00.png".format(filename_base))
    array *= 1.0

    print(dir(cuda_texture_gen))

    erosion = cuda_texture_gen.Erosion3()

    # BEST SO FAR BUT REALLY GETS VERY TALL!?!
    erosion.steps = 512
    erosion.rain_rate = 0.01
    erosion.w_max = 1.0
    erosion.capacity = 0.1 / 10
    erosion.erode = 0.1 / 100
    erosion.deposit = 0.1 / 10
    erosion.evap = 0.1 / 1000
    erosion.wrap = True

    # # test new trying to stop the terrain piling up
    # erosion.steps = 512          # keep moderate
    # erosion.rain_rate = 0.002        # lighter rain
    # erosion.w_max = 0.05         # clamp outflow per step
    # erosion.capacity = 1.0          # allow water to carry sediment
    # erosion.erode = 0.02         # stronger erosion
    # erosion.deposit = 0.01         # weaker than erosion
    # erosion.evap = 0.002        # meaningful evaporation
    # erosion.wrap = True

    # test new trying to stop the terrain piling up again
    erosion.rain_rate = 0.002
    erosion.w_max = 0.05 / 10.0
    erosion.capacity = 1.0 / 2.0
    erosion.erode = 0.02
    erosion.deposit = 0.005   # weaker than erosion
    erosion.evap = 0.002 * 3.0
    erosion.steps = 512 * 2

    print("⛰️", dir(erosion))

    erosion.height_map = array
    erosion.process()

    height_map = erosion.height_map
    water_map = erosion.water_map
    sediment_map = erosion.sediment_map

    print("height_map min: {}, max: {}".format(height_map.min(), height_map.max()))
    print("water_map min: {}, max: {}".format(water_map.min(), water_map.max()))
    print("sediment_map min: {}, max: {}".format(sediment_map.min(), sediment_map.max()))

    normalize_array(water_map)
    normalize_array(height_map)

    save_array_as_image(height_map * 255, "output/{}_01.png".format(filename_base))
    save_array_as_image(water_map * 255, "output/{}_01.water_map.png".format(filename_base))
    save_array_as_image(sediment_map * 255, "output/{}_01.sediment_map.png".format(filename_base))

    cuda_texture_gen.blur(height_map, amount=1.0, wrap=True)
    normalize_array(height_map)
    save_array_as_image(height_map * 255, "output/{}_01.blur.png".format(filename_base))


def test_erosion_3_2():
    function_name = inspect.currentframe().f_code.co_name
    print("⛰️ {}()...".format(function_name))
    filename_base = function_name

    map_width = 1024 * 2

    # array = get_fractal_noise(map_width, map_width, 6, 7)
    # array = get_fractal_noise(map_width, map_width, 2, 7)
    # array = get_fractal_noise(map_width, map_width, 5, 1)
    array = get_fractal_noise(map_width, map_width, 5, 7)

    normalize_array(array)
    save_array_as_image(array * 255, "output/{}_00.png".format(filename_base))
    array *= 1.0

    print(dir(cuda_texture_gen))

    erosion = cuda_texture_gen.Erosion3()

    # BEST SO FAR BUT REALLY GETS VERY TALL!?!
    erosion.steps = 512
    erosion.rain_rate = 0.01
    erosion.w_max = 1.0
    erosion.capacity = 0.1 / 10
    erosion.erode = 0.1 / 100
    erosion.deposit = 0.1 / 10
    erosion.evap = 0.1 / 1000
    erosion.wrap = True

    # # test new trying to stop the terrain piling up
    # erosion.steps = 512          # keep moderate
    # erosion.rain_rate = 0.002        # lighter rain
    # erosion.w_max = 0.05         # clamp outflow per step
    # erosion.capacity = 1.0          # allow water to carry sediment
    # erosion.erode = 0.02         # stronger erosion
    # erosion.deposit = 0.01         # weaker than erosion
    # erosion.evap = 0.002        # meaningful evaporation
    # erosion.wrap = True

    # test new trying to stop the terrain piling up again
    erosion.rain_rate = 0.002
    erosion.w_max = 0.05 / 10.0
    erosion.capacity = 1.0 / 2.0
    erosion.erode = 0.02
    erosion.deposit = 0.005   # weaker than erosion
    erosion.evap = 0.002 * 3.0
    erosion.steps = 512 * 2

    # erosion.steps = 128

    print("⛰️", dir(erosion))

    erosion.height_map = array
    erosion.process()

    height_map = erosion.height_map
    water_map = erosion.water_map
    sediment_map = erosion.sediment_map

    print("height_map min: {}, max: {}".format(height_map.min(), height_map.max()))
    print("water_map min: {}, max: {}".format(water_map.min(), water_map.max()))
    print("sediment_map min: {}, max: {}".format(sediment_map.min(), sediment_map.max()))

    normalize_array(water_map)
    normalize_array(height_map)

    save_array_as_image(height_map * 255, "output/{}_01.png".format(filename_base))
    save_array_as_image(water_map * 255, "output/{}_01.water_map.png".format(filename_base))
    save_array_as_image(sediment_map * 255, "output/{}_01.sediment_map.png".format(filename_base))

    cuda_texture_gen.blur(height_map, amount=1.0, wrap=True)
    normalize_array(height_map)
    save_array_as_image(height_map * 255, "output/{}_01.blur.png".format(filename_base))


test_erosion_3_2()
