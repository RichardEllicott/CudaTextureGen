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

    gna_base.xxx = 1234

    print(gna_base.xxx)


    output = gna_base.output
    print(output)

    pass

test()
