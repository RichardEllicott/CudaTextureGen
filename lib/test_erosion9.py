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


def test_mode_1():
    """
    layers test
    """
    runner = ErosionRunner()
    # runner.debug = False

    runner.output_preset_01()
    # runner.output_preset_02()
    # runner.output_preset_03()

    erosion = runner.erosion
    erosion.mode = 1

    runner.nearest_neighbor_upscale = 1
    map_width, map_height = 512, 512
    map_width //= runner.nearest_neighbor_upscale
    map_height //= runner.nearest_neighbor_upscale

    # scale vars (increase processing time, do smaller steps)
    scale_stretch = 4

    octaves = 8
    octaves = 3
    octaves = 4
    base_period = 1

    height_map = tools.noise.fractal(width=map_width, height=map_height, octaves=octaves, base_period=base_period)
    height_map *= 32.0
    # height_map *= 8.0

    # # circle cuut
    # circle = tools.arrays.circle(map_width, map_height, map_width // 3)
    # circle = tools.arrays.blur(circle, 16.0)
    # height_map *= circle

    erosion.height_map = height_map

    # layer_map = tools.noise.fractal_rgb(width=map_width, height=map_height, octaves=octaves)
    # runner.layer_map = layer_map

    erosion.rain_rate = 0.0007  # increasing rain rate barely making difference!
    erosion.erosion_rate = 0.03
    erosion.evaporation_rate = 0.0001
    erosion.max_water_outflow = 1.0

    # erosion.drain_at_min_height = True
    erosion.min_height = 0.0
    erosion.drain_rate = 0.01

    # erosion.positive_slope_gradient_cap = 16.0

    # # sediment is affecting the erosion a bit, slowing it down i think
    # erosion.sediment_yield = 0.5
    # erosion.sediment_capacity = 0.5 / 4.0
    # erosion.sediment_capacity = 1.0 # blows up!
    # erosion.deposition_rate = 0.5
    # erosion.deposition_threshold = 0.00001

    erosion.slope_jitter = 1.0 / 8.0
    # erosion.slope_jitter = 32.0

    runner.frame_count = 64
    runner.frame_count *= 4

    runner.image_profiles = {}  # disable image

    runner.steps_per_frame = 16

    # scale stretch
    erosion.rain_rate /= scale_stretch
    # erosion.erosion_rate /= scale_stretch # maybe leave the same as we have less water per frame
    erosion.evaporation_rate /= scale_stretch
    erosion.max_water_outflow /= scale_stretch
    runner.steps_per_frame *= scale_stretch
    erosion.drain_rate /= scale_stretch
    erosion.positive_slope_gradient_cap /= scale_stretch

    # runner.steps_per_frame = 64
    runner.process()


# test_mode_0()
test_mode_1()
