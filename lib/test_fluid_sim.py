"""
testing 

"""

from tools import *

folder = "output"



def test():

    height_map = get_fractal_noise(width=256, height=256)


    save_array_as_image(height_map * 255, "{}/{}.png".format(folder, "test_fluid_sim1"))


    fluid_sim = cuda_texture_gen.FluidSimulation()
    fluid_sim.device_array_2d_2 = height_map


    get_back = fluid_sim.device_array_2d_2

    save_array_as_image(get_back * 255, "{}/{}.png".format(folder, "test_fluid_sim2"))


test()