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
    gen.type = 0

    array = gen.generate(1024, 1024)
    normalize_array(array)

    for i in range(7):
        gen.period += 5
        gen.seed += 1
        array += gen.generate(1024, 1024)

    normalize_array(array)

    return array


def get_fractal_noise(width=1024, height=1024, octaves=7, base_period=6, base_seed=0, gain=0.5):
    gen = cuda_texture_gen.NoiseGenerator()
    gen.type = 0  # Assuming 0 = Perlin or similar

    array = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0

    for i in range(octaves):
        gen.period = base_period + i * 5
        gen.seed = base_seed + i

        layer = gen.generate(width, height)
        normalize_array(layer)

        array += layer * amplitude
        total_amplitude += amplitude
        amplitude *= gain  # Reduce amplitude for higher octaves

    array /= total_amplitude  # Normalize final result

    return array

# trying to be more rocky


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


# test macro template design


# test_template_class()

# testing new system for copying data in and out
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


# generate_noise_and_erode() # MAIN TEST ATM

# array = get_fractal_noise()
# save_array_as_image(array * 255, "output/fractal_noise.png")
# array = erode_array(array)
# save_array_as_image(array * 255, "output/fractal_noise_eroded.png")


def test_noise_offet():

    print("🦍 test_noise_offet...")

    gen = cuda_texture_gen.NoiseGenerator()
    gen.period = 13
    gen.type = 1  # value noise

    array = gen.generate(1024, 1024)
    print("height range: [{}, {}]".format(array.min(), array.max()))
    normalize_array(array)
    save_array_as_image(array * 255, "output/noise_offset1.png")

    gen.x = 0.5
    gen.y = 0.5

    # gen.angle = math.radians(45)

    array = gen.generate(1024, 1024)
    print("height range: [{}, {}]".format(array.min(), array.max()))
    normalize_array(array)
    offset_array(array)

    array = rotate(array, 45, reshape=False, mode='wrap')  # scipy rotate

    save_array_as_image(array * 255, "output/noise_offset2.png")

    # # Apply offset (in pixels), then rotate
    # shifted = shift(noise, shift=(dy, dx), mode='wrap')
    # rotated = rotate(shifted, angle_degrees, reshape=False, mode='wrap')


def test_template_class():
    print(dir(cuda_texture_gen))

    template_class = cuda_texture_gen.TemplateClass()

    print(template_class)
    print(dir(template_class))

    array = get_fractal_noise()
    save_array_as_image(array * 255, "output/fractal_noise.png")

    template_class.test(array)

    save_array_as_image(array * 255, "output/fractal_noise2.png")


# test_noise_offet()


def test_template_class2():
    print("🐧 test_template_class2()...")

    print(dir(cuda_texture_gen))
    template_class2 = cuda_texture_gen.TemplateClass2()
    print(dir(template_class2))

    array = get_fractal_noise()
    save_array_as_image(array * 255, "output/test_template_class2_0.png")

    # template_class2.height_map = array
    # template_class2.process()

    template_class2.process(array)

    array = template_class2.height_map

    save_array_as_image(array * 255, "output/test_template_class2_1.png")


# test_template_class2()


def test_test_class_3():
    print("🐮 test_test_class_3()...")

    print(dir(cuda_texture_gen))
    template_class_3 = cuda_texture_gen.TemplateClass3()
    print(dir(template_class_3))

    array = get_fractal_noise()
    save_array_as_image(array * 255, "output/template_class_3_0.png")

    template_class_3.image = array
    template_class_3.process()
    array = template_class_3.image

    save_array_as_image(array * 255, "output/template_class_3_1.png")


def generate_noise_and_erode():

    print("⛰️")

    gen = cuda_texture_gen.NoiseGenerator()

    gen.period = 14
    gen.seed = 0

    print(dir(gen))

    array = gen.generate(1024, 1024)

    print("height range: [{}, {}]".format(array.min(), array.max()))

    normalize_array(array)

    save_array_as_image(array * 255, "output/noise.png")

    # erosion = cuda_texture_gen.Erosion()
    erosion = cuda_texture_gen.Erosion2()

    print(dir(erosion))

    # WORKING DEFAULTS
    erosion.mode = 2
    erosion.flow_factor = 0.2
    erosion.erosion_rate = 0.01
    erosion.deposition_rate = 0.02 * 0.5
    erosion.slope_threshold = 0.01

    erosion.rain_rate = 0.01

    # erosion.jitter = 0.01
    erosion.wrap = True
    erosion.steps = 512 * 2

    erosion.process(array)
    array = erosion.height_map
    offset_array(array)  # test seamless
    save_array_as_image(array * 255, "output/erosion.png")

    save_array_as_image(erosion.sediment_map * 255, "output/sediment_map.png")
    save_array_as_image(erosion.water_map * 255, "output/water_map.png")


# generate_noise_and_erode()


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


def test_blur(filename, output_filename, amount=1):
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    print("load image...")
    img = Image.open(filename).convert("L")
    arr = np.array(img, dtype=np.float32)
    print("height range: [{}, {}]".format(arr.min(), arr.max()))
    print("blur...")

    cuda_texture_gen.blur(arr, amount=1, wrap=True)

    save_array_as_image(arr, output_filename)

# LICENSE 
def test_resample():

    # noise = get_fractal_noise(width=1024, height=1024, octaves=6, base_period=2, base_seed=0, gain=0.8, lacunarity=2.0)
    noise = get_fractal_noise(width=1024, height=1024, octaves=3, base_period=7, base_seed=0, gain=0.8, lacunarity=1.9)
    normalize_array(noise)

    save_array_as_image(noise * 255, "output/00_noise.png")

    gen = cuda_texture_gen.NoiseGenerator()
    gen.period = 5
    gen.seed = 0

    map_x = gen.generate(1024, 1024)
    normalize_array(map_x)
    save_array_as_image(map_x * 255, "output/01_map_x.png")

    gen.seed += 1
    map_y = gen.generate(1024, 1024)
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

    resample = cuda_texture_gen.Resample()

    resample.input = noise

    resample.map_x = map_x * 64
    resample.map_y = map_y * 64

    resample.process()

    resampled = resample.output
    save_array_as_image(resampled * 255, "output/03_resampled.png")

    erosion = cuda_texture_gen.Erosion2()

    erosion.height_map = resampled
    erosion.steps = 64 * 4
    erosion.process()

    eroded = erosion.height_map
    save_array_as_image(eroded * 255, "output/04_eroded.png")

    shader_maps = cuda_texture_gen.ShaderMaps

    print(shader_maps)

    test_shader_maps("output/04_eroded.png")

    # normal_map = shader_maps.generate_normal_map(eroded, 1.0, True)

    # normal_map = shader_maps.generate_normal_map(eroded)
    # ao_map = shader_maps.generate_ao_map(eroded * 0.5, 2.0, True)

    # print("Type:", type(eroded))
    # print("Dtype:", eroded.dtype)
    # print("Shape:", eroded.shape)
    # print("Flags:", eroded.flags)

    # arr = np.array(eroded * 0.5, dtype=np.float32)
    # print("Final Type:", type(arr))
    # print("Final Dtype:", arr.dtype)
    # print("Final Flags:", arr.flags)

    # ao_map = shader_maps.generate_ao_map(eroded * 0.5, 2, True)
    # ao_map = shader_maps.generate_ao_map(resampled)

    # save_array_as_image(normal_map * 255, "output/05_normal_map.png")
    # save_array_as_image(ao_map * 255, "output/06_ao_map.png")

    test_blur("output/04_eroded.ao.png", "output/04_eroded.ao.blur.png")

# test_resample()


print(dir(cuda_texture_gen))
template_base_1_test = cuda_texture_gen.TemplateBase1Test()
print(dir(template_base_1_test))
template_base_1_test.process()

