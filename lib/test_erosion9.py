"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""
from matplotlib.colors import to_rgb
import tools
import cuda_texture_gen
import numpy as np

from ErosionRunner import ErosionRunner
from IslandGenerator import IslandGenerator


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
    erosion.drain_rate = 100000.0

    runner.frame_count = 256


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


def test_mode_1():
    """
    layers test
    """
    runner = ErosionRunner()
    erosion = runner.erosion
    island_generator = IslandGenerator()

    runner.output_preset_01()
    # runner.output_preset_02()
    # runner.output_preset_03()

    # runner.image_profiles = {}  # disable image
    runner.movie_profiles = {}  # disable movies

    map_size = 1024

    erosion.mode = 1

    runner.nearest_neighbor_upscale = 1
    map_width = map_size
    map_height = map_size


    map_width //= runner.nearest_neighbor_upscale
    map_height //= runner.nearest_neighbor_upscale

    height_scale = 64.0
    height_scale = 24.0
    scale_stretch = 1 # stretching time

    # gen heightmaps
    octaves = 8
    octaves = 10
    base_period = 1
    height_map = tools.noise.fractal(width=map_width, height=map_height, octaves=octaves, base_period=base_period)

    cut_island = True
    if cut_island:
        # island cut
        island_generator.width = map_width
        island_generator.height = map_height
        island_generator.preset00()
        island_generator.pre_blur = 32.0
        island = island_generator.island
        height_map *= island
        # plus adding a bit of island height back
        # height_map += island * 2.0
        # tools.arrays.normalized(height_map)

    # erosion pars
    erosion.rain_rate = 0.0007  # increasing rain rate barely making difference!
    erosion.erosion_rate = 0.01
    erosion.evaporation_rate = 0.0002
    erosion.min_height = 0.0
    # erosion.drain_rate = 0.001
    erosion.slope_jitter = 1.0

    # erosion.flow_rate = 0.125
    erosion.max_water_outflow = 0.125

    # sediment
    erosion.sediment_yield = 0.5
    erosion.sediment_capacity = 1.0
    # # erosion.deposition_mode = 1
    # # erosion.deposition_rate = 0.125
    # # erosion.deposition_threshold = 0.125

    # runner
    runner.frame_count = 256
    runner.frame_count = 128
    runner.steps_per_frame = 16



    # scale stretch (if used)
    erosion.rain_rate /= scale_stretch
    # erosion.erosion_rate /= scale_stretch # maybe leave the same as we have less water per frame
    erosion.evaporation_rate /= scale_stretch
    erosion.max_water_outflow /= scale_stretch
    runner.steps_per_frame *= scale_stretch
    erosion.drain_rate /= scale_stretch
    erosion.positive_slope_gradient_cap /= scale_stretch

    erosion.height_map = height_map * height_scale

    runner.process()


# test_mode_0()
test_mode_1()
