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
from IslandGenerator import IslandGenerator

from pathlib import Path

import math



runner = ErosionRunnerGNC()
erosion = runner.erosion
island_generator = IslandGenerator()

script_path = Path(__file__)


def test():

    # ================================================================
    gnc_noise = ct.GNC_Noise()
    gnc_noise.period = (13, 13, 13)
    gnc_noise.wrap = (False, False, False)
    gnc_noise.size = (256, 256)
    gnc_noise.mode = 1

    # ----------------------------------------------------------------

    gnc_noise.process()
    noise = gnc_noise.output.array
    # tools.arrays.offset(noise)
    tools.arrays.normalize(noise)
    tools.images.save(noise, f"{script_path}.noise.png")



    gnc_noise.rotation = (0.0, 0.0, math.radians(20))
    gnc_noise.process()
    noise = gnc_noise.output.array
    tools.arrays.normalize(noise)

    # tools.arrays.rotate(noise, math.radians(-90))

    tools.images.save(noise, f"{script_path}.noise2.png")


    # ================================================================
    # noise = tools.noise.fractal(256, 256)
    # tools.images.save(noise, f"{script_path}.noise.png")
    # ================================================================

    # DEBUG

    return


    device_array = ct.DeviceArrayFloat2D()
    device_array.array = noise

    erosion.steps = 512
    erosion.rain_rate = 0.0007
    erosion.erosion_rate = 0.001


    erosion.layer_erosiveness_array = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ]

    print(erosion.height_map)
    erosion.height_map = device_array

    print(erosion.height_map)

    # runner.process() # running process now messes up the next part
    erosion.compute()

    # erosion

    erosion.stream.sync()  # ⚠️

    output = erosion.height_map.array
    tools.arrays.normalize(output)

    tools.images.save(output, f"{script_path}.erode.png")


test()
