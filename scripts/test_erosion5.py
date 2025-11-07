"""
testing 

"""

from tools import *


print(dir(cuda_texture_gen))


def test_erosion5():

    # erosion = cuda_texture_gen.Erosion3() # the old working but it has the wrong order of stuff!!
    erosion = cuda_texture_gen.Erosion5()

    erosion.steps = 1024

    erosion.rain_rate = 0.001
    erosion.rain_rate = 0.0


    erosion.max_water_outflow = 1000


    folder = "output"
    base_filename = "test_erosion5"

    terrain = get_fractal_noise(octaves = 3)
    normalize_array(terrain)


    start_water = get_fractal_noise(octaves = 3, base_seed=2567)
    normalize_array(start_water)
    erosion.water_map = start_water


    save_array_as_image(terrain * 255, "{}/{}.png".format(folder, base_filename + ".terrain"))

    save_array_as_image(start_water * 255, "{}/{}.png".format(folder, base_filename + ".start_water"))


    erosion.height_map = terrain
    erosion.process()
    eroded = erosion.height_map

    # print("ERODED HEIGHT min: {}, max: {}".format(array.min(), array.max()))

    normalize_array(eroded)

    save_array_as_image(eroded * 255, "{}/{}.png".format(folder, base_filename + ".eroded"))



    

    # normal_map = generate_normal_map(array, strength=4.0)
    # save_array_as_image(normal_map * 255, "{}/{}.png".format(folder, base_filename + ".erode.normal"))
    # ao_map = generate_ao_map(array * 64, radius=3.0, mode=0)
    # save_array_as_image(ao_map * 255, "{}/{}.png".format(folder, base_filename + ".erode.ao"))
    # albedo = apply_color_map(array)
    # save_array_as_image(albedo * 255, "{}/{}.png".format(folder, base_filename + ".albedo"))

    water_map = erosion.water_map
    blur(water_map, 0.5)
    save_array_as_image(water_map * 255, "{}/{}.png".format(folder, base_filename + ".water.blur"))


    save_array_as_image(erosion.sediment_map * 255, "{}/{}.png".format(folder, base_filename + ".sediment"))
    save_array_as_image(erosion.water_map * 255, "{}/{}.png".format(folder, base_filename + ".water"))


test_erosion5()
