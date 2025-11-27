"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""


from matplotlib.colors import to_rgb
import tools
import cuda_texture_gen
import numpy as np


from ErosionRunner import ErosionRunner


# --------print--------
def get_heightmap_01(width=128, height=128):

    octaves = 7
    # octaves = 4 + 1
    # heightmap_scale = 16.0 * 4

    height_map = tools.fractal_noise(width=width, height=height, octaves=octaves)
    tools.normalize_array(height_map)

    return height_map


# 2D array with shape (0, 0) .... USE FOR CLEARING A HEIGHTMAP.. OPTIONAL
empty_map = np.empty((0, 0), dtype=np.float32)


def test01():
    """
    shows okay rivers
    """
    height_map = tools.fractal_noise(width=256, height=256, octaves=5)
    tools.normalize_array(height_map)

    runner = ErosionRunner()
    runner.PRESET_erosion_01()
    # runner.PRESET_erosion_02()
    # erosion.PRESET_simple_erosion()

    runner.nearest_neighbor_upscale = 2
    runner.erosion.height_map = height_map * 128.0
    # runner.erosion.height_map = get_heightmap_01() * 32.0

    runner.process()

# test()


def lerp(a, b, t: float):
    return (1.0 - t) * a + t * b


def get_island_height_map(width, height):

    island = tools.load_image_as_array("./images/island03.png")
    island = tools.resize_array_2d(island, width, height)
    island = tools.blur_array_2d(island, 8)

    # tools.normalize_array(island)

    octaves = 8
    noise = tools.fractal_noise(width=width, height=height, octaves=octaves)
    tools.normalize_array(noise)

    fade = 0.5

    result = island * noise

    # result = lerp(result, noise, 0.7)

    tools.save_array_as_image(result, "./images/island03.out.png")

    return result


# get_island_height_map(512, 512)


def test02():
    """
    shows okay rivers
    """
    runner = ErosionRunner()
    erosion = runner.erosion

    height_map = empty_map
    water_map = empty_map
    sediment_map = empty_map
    rain_map = empty_map
    ################################################################
    map_width, map_height = 1024, 1024
    map_width, map_height = 512, 512

    # map_width, map_height = 256, 256
    # runner.nearest_neighbor_upscale = 2

    # map_width, map_height = 128, 128
    # runner.nearest_neighbor_upscale = 4
    ################################################################
    octaves = 8
    # octaves = 2
    # octaves = 4

    height_map = tools.fractal_noise(width=map_width, height=map_height, octaves=octaves)
    # height_map = tools.normalized_array(height_map)

    # height_map = get_island_height_map(map_width, map_height)

    # rain_map = height_map.copy() # rain map based on height
    # rain_map *= 128.0 #❗got a strange result with no copy
    # rain_map = rain_map / 2.0 + 0.5

    # rain_map = 1.0 - rain_map # invert
    height_map *= 128.0

    # add water
    erosion.rain_rate = 0.005

    erosion.max_water_outflow = 1.0
    erosion.erosion_mode = 1
    # erosion.erosion_mode = 2
    erosion.deposition_rate = 0.5 / 10.0
    # erosion.deposition_rate = 0.5
    # erosion.deposition_rate = 0.9
    erosion.erosion_rate = 0.1 / 4.0
    # erosion.erosion_rate = 0.2

    # remove water
    erosion.evaporation_rate = 0.001
    erosion.drain_at_min_height = True
    # erosion.min_height = 0.0

    # trying to spread water
    # self._erosion.diffusion_rate = 0.001 # seems buggy
    # self._erosion.max_water_outflow = 0.2

    # changed_pars = dict_changes(default_pars, object_pars_to_dict(erosion))
    # print("🏔️", changed_pars)

    ################################################################
    erosion.height_map = height_map
    erosion.water_map = water_map
    erosion.sediment_map = sediment_map
    erosion.rain_map = rain_map

    runner.steps_per_frame = 16
    runner.steps_per_frame = 64
    runner.process()


# test02()s





def test_3d_arrays():

    soil_colors = np.array([
        to_rgb("#8b4513"),  # [139/255, 69/255, 19/255] → saddle brown
        to_rgb("#a0522d"),  # [160/255, 82/255, 45/255] → sienna
        to_rgb("#d2b48c"),  # [210/255, 180/255, 140/255] → tan
    ], dtype=np.float32)

    noise = tools.fractal_noise_rgb(256, 256).astype(np.float32)  # expect [0,1] floats

    den = noise.sum(axis=-1, keepdims=True)
    weights = noise / np.clip(den, 1e-8, None)

    noise_to_pallete = weights @ soil_colors  # matrix multiply (256, 256, 3)

    # island = cuda_texture_gen.blur

    tools.save_array_as_image((noise_to_pallete * 255).astype(np.uint8), "./output/noise.pallete.png")

    # noise = np.ascontiguousarray(noise[:, :, 0], dtype=np.float32)
    noise = noise[:, :, 0]
    print("🐈 ", type(noise))
    print("shape:", noise.shape)        # dimensions
    print("dtype:", noise.dtype)        # element type
    print("ndim:", noise.ndim)          # number of dimensions
    print("strides:", noise.strides)    # byte steps between elements
    print("flags:", noise.flags)        # contiguity info

    # noise = tools.fractal_noise(256, 256)s
    # print("🐈 ", type(noise))
    # print("shape:", noise.shape)        # dimensions
    # print("dtype:", noise.dtype)        # element type
    # print("ndim:", noise.ndim)          # number of dimensions
    # print("strides:", noise.strides)    # byte steps between elements
    # print("flags:", noise.flags)        # contiguity info

    cuda_texture_gen.blur(noise, 32)

    tools.save_array_as_image((noise * 255).astype(np.uint8), "./output/noise.rgb.png")


# test_3d_arrays()

def test_3d_arrays():

    # noise = tools.fractal_noise_rgb(256, 256).astype(np.float32)  # expect [0,1] floats
    noise = tools.fractal_noise(256, 256).astype(np.float32)  # expect [0,1] floats

    runner = ErosionRunner()
    erosion = runner.erosion

    erosion.layered_height_map = noise

    noise = erosion.layered_height_map

    tools.print_array_information(noise)


# test_3d_arrays()

def layers_test():
    runner = ErosionRunner()
    erosion = runner.erosion
    # print("layers_name:", erosion.layers_name)
    print("layers_resistance:", erosion.layers_resistance)
    print("layers_yield:", erosion.layers_yield)
    print("layers_permeability:", erosion.layers_permeability)
    print("layers_threshold:", erosion.layers_threshold)


# layers_test()
