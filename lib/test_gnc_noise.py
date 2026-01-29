"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""
from PIL import Image
# from matplotlib.colors import to_rgb
import tools
# import numpy as np
import cuda_texture_gen as ct


# from ErosionRunner import ErosionRunnerGNC
# from IslandGenerator import IslandGenerator

from pathlib import Path

# import math



# runner = ErosionRunnerGNC()
# erosion = runner.erosion
# island_generator = IslandGenerator()

script_path = Path(__file__)



output_folder: str = "./godot/cuda_texture_gen/projects/scroll_shader/"

def gen_noise():

    print("gen_noise()...")

    map_size = (512, 512)


    for i in range(4):

        noise = tools.noise.fractal(map_size[0], map_size[1], octaves=13, base_period=5, seed=i)
        tools.arrays.normalize(noise)
        tools.images.save(noise, f"{output_folder}noise.{i:02d}.png")



gen_noise()