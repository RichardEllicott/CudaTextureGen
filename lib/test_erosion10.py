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


def test_mode_1():
    """
    layers test
    """
    runner = ErosionRunner()
    erosion = runner.erosion
    island_generator = IslandGenerator()

    # runner.output_preset_01()
    runner.output_preset_02()
    # runner.output_preset_03()

    runner.image_profiles = {}  # disable images
    # runner.movie_profiles = {}  # disable movies

    map_size = 1024 // 2

    # erosion.mode = 1  # 🧪
    # erosion.mode = 2  # 🧪

    runner.nearest_neighbor_upscale = 1
    map_width = map_size
    map_height = map_size

    map_width //= runner.nearest_neighbor_upscale
    map_height //= runner.nearest_neighbor_upscale

    height_scale = 64.0
    # height_scale = 24.0
    scale_stretch = 1  # stretching time

    # gen heightmaps
    octaves = 8
    # octaves = 5
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
    erosion.rain_rate = 0.0007 * 10.0
    erosion.erosion_rate = 0.01 * 10.0
    erosion.evaporation_rate = 0.0002
    erosion.min_height = 0.0
    # erosion.drain_rate = 0.001
    erosion.slope_jitter = 1.0
    # erosion.slope_jitter_mode = 1

    # erosion.flow_rate = 0.125
    erosion.max_water_outflow = 0.125
    erosion.max_water_outflow = 1.0

    # sediment

    sediment_tests = False
    # sediment_tests = True
    if sediment_tests:
        # erosion.deposition_mode = 1
        erosion.sediment_yield = 0.125
        erosion.sediment_capacity = 0.5
        erosion.deposition_rate = 0.5
        # erosion.deposition_threshold = 0.125

    # runner
    runner.frame_count = 256
    # runner.frame_count = 128
    runner.steps_per_frame = 16
    # runner.steps_per_frame = 64

    # scale stretch (if used)
    erosion.rain_rate /= scale_stretch
    # erosion.erosion_rate /= scale_stretch # maybe leave the same as we have less water per frame
    erosion.evaporation_rate /= scale_stretch
    erosion.max_water_outflow /= scale_stretch
    runner.steps_per_frame *= scale_stretch
    erosion.drain_rate /= scale_stretch
    # erosion.positive_slope_gradient_cap /= scale_stretch

    erosion.height_map = height_map * height_scale

    runner.process()


# test_mode_0()
test_mode_1()
