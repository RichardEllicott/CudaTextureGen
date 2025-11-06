"""
testing 

"""

from tools import *


print(dir(cuda_texture_gen))


def test_erosion5():

    # erosion = cuda_texture_gen.Erosion5()
    erosion = cuda_texture_gen.Erosion3() # the old working one TEST





    folder = "output"
    # folder = "godot/cuda_texture_gen/textures/animation_test"
    base_filename = "test_erosion5"

    array = get_fractal_noise()
    print("START HEIGHT min: {}, max: {}".format(array.min(), array.max()))
    normalize_array(array)

    save_array_as_image(array * 255, "{}/{}.png".format(folder, base_filename + ".noise"))

    erosion.steps = 1024

    erosion.height_map = array
    erosion.process()
    array = erosion.height_map
    print("ERODED HEIGHT min: {}, max: {}".format(array.min(), array.max()))

    normalize_array(array)

    save_array_as_image(array * 255, "{}/{}.png".format(folder, base_filename + ".erode"))

    normal_map = generate_normal_map(array, strength=4.0)
    save_array_as_image(normal_map * 255, "{}/{}.png".format(folder, base_filename + ".erode.normal"))

    ao_map = generate_ao_map(array * 64, radius=3.0, mode=0)
    save_array_as_image(ao_map * 255, "{}/{}.png".format(folder, base_filename + ".erode.ao"))



    albedo = apply_color_map(array)
    save_array_as_image(albedo * 255, "{}/{}.png".format(folder, base_filename + ".albedo"))


    array = erosion.sediment_map
    print("min: {}, max: {}".format(array.min(), array.max()))
    save_array_as_image(array * 255, "{}/{}.png".format(folder, base_filename + ".sediment"))


test_erosion5()
