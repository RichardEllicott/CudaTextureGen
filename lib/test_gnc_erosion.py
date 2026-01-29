"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""
from PIL import Image
from matplotlib.colors import to_rgb
import tools
import numpy as np
import cuda_texture_gen as ct


from ErosionRunner import ErosionRunnerGNC
from ErosionRunner import ErosionRunnerDelta

from IslandGenerator import IslandGenerator

from pathlib import Path

import math


# runner = ErosionRunnerGNC()
runner = ErosionRunnerDelta()


erosion = runner.erosion
island_generator = IslandGenerator()

script_path = Path(__file__)


def test_runner():
    print("test_runner()...")


    map_size = (512, 512)
    # map_size = (1024, 1024)

    # ================================================================
    gnc_noise = ct.GNC_Noise()
    gnc_noise.period = (20, 13, 13)
    gnc_noise.wrap = (False, False, False)
    gnc_noise.size = map_size
    gnc_noise.mode = 1
    # ----------------------------------------------------------------
    gnc_noise.compute()
    # noise = gnc_noise.output.array
    # tools.arrays.offset(noise)
    # tools.arrays.normalize(noise)
    # tools.images.save(noise, f"{script_path}.noise.png")
    # ----------------------------------------------------------------

    # gnc_noise.rotation = (0.0, 0.0, math.radians(20))
    # gnc_noise.process()
    # noise = gnc_noise.output.array
    # tools.arrays.normalize(noise)
    # # tools.arrays.rotate(noise, math.radians(-90))
    # tools.images.save(noise, f"{script_path}.noise2.png")

    # ================================================================
    noise = tools.noise.fractal(map_size[0], map_size[1], octaves=7, base_period=1)
    tools.arrays.normalize(noise)
    noise *= 24.0

    # tools.images.save(noise, f"{script_path}.noise.png")
    # ================================================================


    ISLAND_MODE = True
    if ISLAND_MODE:
        # island cut
        island_generator.width = map_size[0]
        island_generator.height = map_size[1]
        island_generator.preset00()
        island_generator.pre_blur = 32.0
        island = island_generator.island
        noise *= island

        # for layer in layers:
            # layer *= island

    # DEBUG

    # return

    runner.OUTPUT_PRESET_02()

    device_array = ct.DeviceArrayFloat2D()
    device_array.array = noise

    erosion.steps = 16
    erosion.rain_rate = 0.0007

    runner.frame_count = 256

    
    erosion.erosion_rate = 0.001 / 4.0
    # erosion.erosion_rate = 0.005
    
    erosion.slope_jitter = 1.0
    erosion.max_water_outflow = 0.125
    erosion.max_water_outflow = 1.0

    # erosion.sediment_yield = 0.5
    # erosion.sediment_capacity = 1.0

    # erosion.erosion_mode = 4

    erosion.min_height = 0.0
    erosion.drain_rate = 1.0 / 1000.0

    erosion.layer_erosiveness_array = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    print(erosion.height_map)
    erosion.height_map = device_array

    print(erosion.height_map)

    runner.process()  # running process now messes up the next part
    # erosion.compute()

    # erosion

    # erosion.stream.sync()  # ⚠️

    # output = erosion.height_map.array
    # tools.arrays.normalize(output)

    # tools.images.save(output, f"{script_path}.erode.png")


test_runner()

