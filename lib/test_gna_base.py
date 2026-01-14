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
from pathlib import Path


script_path = Path(__file__)

# these five this actual script
script_dir = Path(__file__).resolve().parent
icon_path = f"{script_dir}/node_editor_qt.icon.png"

script_filename = Path(__file__).name
scrip_stem = Path(__file__).stem  # minus the extension


def get_test_gradient(width=128, height=128):
    return np.linspace(0.0, 1.0, height*width, dtype=np.float32).reshape((height, width))  # Create a 2D float32 array gradient in range [0, 1]


def test_gna():
    print("test()...")

    test_gradient = get_test_gradient()
    tools.images.save(test_gradient, f"{script_path}.png")

    gna_base = cuda_texture_gen.GNA_Example()

    # test class property
    gna_base.test = 1234
    print(f"gna_base.test = {gna_base.test}")

    gna_base.tile_size = 8

    # test class array
    print(f"input = {gna_base.input}")

    device_array_2d = cuda_texture_gen.DeviceArrayFloat2D()
    # device_array_2d.resize([32, 32])

    device_array_2d.array = test_gradient

    print(f"device_array_2d = {device_array_2d}")
    gna_base.input = device_array_2d
    print(f"gna_base.input = {gna_base.input}")

    gna_base.process()

    tools.images.save(gna_base.input.array, f"{script_path}.input.png")
    tools.images.save(gna_base.output.array, f"{script_path}.output.png")


def test_gnb():
    # print("test()...")

    # test_gradient = get_test_gradient()
    # tools.images.save(test_gradient, f"{script_path}.png")

    # gna_base = cuda_texture_gen.GNB_Example()

    # # test class property
    # gna_base.test = 1234
    # print(f"gna_base.test = {gna_base.test}")

    # gna_base.tile_size = 8

    # # test class array
    # print(f"input = {gna_base.input}")

    # device_array_2d = cuda_texture_gen.DeviceArrayFloat2D()
    # # device_array_2d.resize([32, 32])

    # device_array_2d.array = test_gradient

    # print(f"device_array_2d = {device_array_2d}")
    # gna_base.input = device_array_2d
    # print(f"gna_base.input = {gna_base.input}")

    # gna_base.process()

    # tools.images.save(gna_base.input.array, f"{script_path}.input.png")
    # tools.images.save(gna_base.output.array, f"{script_path}.output.png")
    pass

# test_gnb()


import math

def test_noise():
    print("test()...")

    gnc = cuda_texture_gen.GNC_Noise()

    gnc.period = (13, 13, 5)
    # gnc.size = (512, 512)

    # gnc.smoothing_mode = 0
    # gnc.smoothing_mode = 4

    gnc.process()

    # gnc.rotation = (math.radians(45), math.radians(45), math.radians(45))

    result = gnc.output.array

    tools.arrays.offset(result)

    tools.arrays.normalize(result)
    tools.images.save(result, f"{script_path}.output.png")


test_noise()


# def test_new_erosion():
#     print("test_new_erosion()...")

#     erosion = cuda_texture_gen.GNC_Erosion()

#     erosion.process()

#     # result = gnc.output.array
#     # tools.arrays.normalize(result)
#     # tools.images.save(result, f"{script_path}.output.png")
