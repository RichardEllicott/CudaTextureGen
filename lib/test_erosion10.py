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


runner = ErosionRunner()
erosion = runner.erosion
island_generator = IslandGenerator()


def test_mode_2():
    """
    layers test
    """


    LAYER_MODE = False
    ISLAND_MODE = False
    # LAYER_MODE = True
    ISLAND_MODE = True

    # new simple erode test
    TEST_SIMPLE_ERODE = False
    # TEST_SIMPLE_ERODE = True
    if TEST_SIMPLE_ERODE:
        erosion._main_loop = False
        erosion.simple_collapse = True
        erosion.simple_collapse_amount = 1.0 / 16.0
        erosion.simple_collapse_threshold = 0.0
        erosion.simple_collapse_yield = 0.0
        erosion.simple_collapse_yield = 0.5

    runner.OUTPUT_PRESET_01()
    # runner.OUTPUT_PRESET_02()
    # runner.OUTPUT_PRESET_03()
    # runner.OUTPUT_PRESET_add_layers_01() #WILL CRASH WITH NORMALIZE???

    # runner.image_profiles = {}  # disable images
    # runner.movie_profiles = {}  # disable movies


    map_size = 1024 // 2

    # erosion.erosion_mode = 1  # 🧪 WEIRD SQUARE RESULT
    # erosion.erosion_mode = 2  # 🧪 WEIRD SQUARE RESULT
    # erosion.erosion_mode = 3  # 🧪 less square?
    # erosion.erosion_mode = 4  # 🧪 soft saturation based

    # erosion pars
    erosion.rain_rate = 0.007
    erosion.erosion_rate = 1.0 / 8.0 / 4.0
    erosion.evaporation_rate = 0.002
    erosion.min_height = 0.0
    # erosion.drain_rate = 0.001
    erosion.slope_jitter = 1.0
    # erosion.slope_jitter_mode = 1

    runner.nearest_neighbor_upscale = 1
    map_width = map_size
    map_height = map_size

    map_width //= runner.nearest_neighbor_upscale
    map_height //= runner.nearest_neighbor_upscale

    # height_scale = 64.0
    height_scale = 24.0
    scale_stretch = 1  # stretching time

    # gen heightmaps
    # octaves = 8
    octaves = 7
    base_period = 1

    # layers
    layers = [tools.noise.fractal(width=map_width, height=map_height, octaves=octaves, base_period=base_period, seed=0)]
    if LAYER_MODE:
        for i in range(2):
            height_map = tools.noise.fractal(width=map_width, height=map_height, octaves=octaves, base_period=base_period, seed=i+1)
            layers.append(height_map)

    if ISLAND_MODE:
        # island cut
        island_generator.width = map_width
        island_generator.height = map_height
        island_generator.preset00()
        island_generator.pre_blur = 32.0
        island = island_generator.island

        for layer in layers:
            layer *= island

        # plus adding a bit of island height back
        # height_map += island * 2.0
        # tools.arrays.normalized(height_map)

    # erosion.flow_rate = 0.125
    # erosion.max_water_outflow = 1.0 / 8.0
    # erosion.max_water_outflow = 1.0

    # sediment

    sediment_tests = False
    # sediment_tests = True
    if sediment_tests:
        # erosion.deposition_mode = 1
        erosion.sediment_yield = 0.125 / 8.0
        erosion.sediment_capacity = 0.5
        erosion.deposition_rate = 0.5
        # erosion.deposition_threshold = 0.125

    # runner
    runner.frame_count = 256
    # runner.frame_count = 128
    runner.steps_per_frame = 1
    # runner.steps_per_frame = 64

    # scale stretch (if used)
    erosion.rain_rate /= scale_stretch
    # erosion.erosion_rate /= scale_stretch # maybe leave the same as we have less water per frame
    erosion.evaporation_rate /= scale_stretch
    erosion.max_water_outflow /= scale_stretch
    runner.steps_per_frame *= scale_stretch
    erosion.drain_rate /= scale_stretch
    # erosion.positive_slope_gradient_cap /= scale_stretch

    if LAYER_MODE:

        assert (len(layers) == 3)
        # Merge into one 3D array (height x width x channels)
        rgb = np.stack(layers, axis=-1)
        rgb *= height_scale

        # tools.arrays.print_array_information(rgb)

        erosion.layer_map = rgb

        # # Split back into separate channels (EXAMPLE)
        # R2, G2, B2 = np.dsplit(rgb, 3)   # each has shape (100, 100, 1)
        # R2 = R2.squeeze()
        # G2 = G2.squeeze()
        # B2 = B2.squeeze()

    else:
        erosion.height_map = layers[0] * height_scale

    
def test_mode_2_mod():


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


    erosion.erosion_mode = 4


test_mode_2()

test_mode_2_mod()



runner.process()


