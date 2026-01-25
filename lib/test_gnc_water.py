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





def test_water():

    print("test_water()...")


    water = ct.GNC_Water()
    # template = ct.GNC_Template()


    water.test()


    


test_water()