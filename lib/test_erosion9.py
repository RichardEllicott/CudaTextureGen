"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""
from matplotlib.colors import to_rgb
import tools
import cuda_texture_gen
import numpy as np

from ErosionRunner import ErosionRunner


# 2D array with shape (0, 0) .... USE FOR CLEARING A HEIGHTMAP.. OPTIONAL
empty_map = np.empty((0, 0), dtype=np.float32)


def lerp(a, b, t: float):
    return (1.0 - t) * a + t * b


def test_mode_0():
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

    height_map = tools.noise.fractal(width=map_width, height=map_height, octaves=octaves)
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
    erosion.mode = 0
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

    erosion.sediment_capacity = 1.0  # ⚠️  new change

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


# test_mode_0()

def test_mode_1():
    """
    layers test
    """
    runner = ErosionRunner()
    erosion = runner.erosion
    erosion.mode = 1

    map_width, map_height = 128, 128

    octaves = 8

    height_map = tools.noise.fractal(width=map_width, height=map_height, octaves=octaves)
    height_map *= 32.0

    # ## circle cuut
    # circle = tools.arrays.circle_array(map_width, map_height, map_width // 3)
    # circle = tools.blur_array(circle, 16.0)
    # height_map *= circle


    erosion.height_map = height_map

    # layer_map = tools.noise.fractal_rgb(width=map_width, height=map_height, octaves=octaves)
    # runner.layer_map = layer_map


    erosion.rain_rate = 0.001 # increasing rain rate barely making difference!
    erosion.erosion_rate = 0.01
    erosion.evaporation_rate = 0.0001

    runner.nearest_neighbor_upscale = 4


    # erosion.drain_at_min_height = True
    # erosion.min_height = 0.0
    erosion.max_water_outflow = 1.0


    runner.frame_count = 64


    runner.image_profiles = None # disable image

    runner.steps_per_frame = 16
    # runner.steps_per_frame = 64
    runner.process()


test_mode_1()
