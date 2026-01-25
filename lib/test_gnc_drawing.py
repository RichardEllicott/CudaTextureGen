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





def test_drawing():

    print("test_drawing()...")


    drawing = ct.GNC_Drawing()
    # template = ct.GNC_Template()

    print("test_drawing()...")

    # drawing.radius /= 2.0


    drawing.size = (64, 64)
    drawing.radius = drawing.size[0] / 2.0
    drawing.position = (drawing.size[0] / 2, drawing.size[1] / 2)

    for i in range(5):

        drawing.mode = i
        drawing.compute()
        tools.images.save(drawing.output.array, f"{script_path}.sample.{i}.png")



    


test_drawing()