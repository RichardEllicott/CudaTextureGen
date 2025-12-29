"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""
from matplotlib.colors import to_rgb
import tools
import cuda_texture_gen
import numpy as np


import numpy as np
import cuda_texture_gen


def test():
    print("test()...")

    gna_base = cuda_texture_gen.GNA_Base()

    # test class property
    gna_base.test = 1234
    print(f"gna_base.test = {gna_base.test}")

    # test class array
    print(f"input = {gna_base.input}")

    device_array_2d = cuda_texture_gen.DeviceArrayFloat2D()
    device_array_2d.resize([32, 32])


    print(f"device_array_2d = {device_array_2d}")
    gna_base.input = device_array_2d
    print(f"gna_base.input = {gna_base.input}")


    gna_base.process()

    pass


test()
