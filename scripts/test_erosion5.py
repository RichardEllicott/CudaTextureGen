"""
testing 

"""

from tools import *
# import numpy as np

folder = "output"


erosion = cuda_texture_gen.Erosion5()
default_pars = object_pars_to_dict(erosion)



# print("default pars: ")
# for key in default_pars:
#     print("erosion.{} = {}".format(key, default_pars[key]))
# print()


def erosion_preset_01() -> None:

    # reflect_object(erosion)

    erosion.steps = 1024 // 2

    erosion.rain_rate = 0.05 * 1.7
    erosion.max_water_outflow = 1.0
    # erosion.erosion_mode = 2
    erosion.erosion_mode = 1
    erosion.deposition_rate = 0.5 / 10.0
    erosion.erosion_rate = 0.1
    erosion.evaporation_rate = 0.03
    erosion.min_height = 0.0

    # trying to spread water
    # erosion.diffusion_rate = 0.001 # seems buggy
    # erosion.max_water_outflow = 0.2

    changed_pars = dict_changes(default_pars, object_pars_to_dict(erosion))
    print("🏔️", changed_pars)

    return erosion


def get_heightmap_01(width=512, height=512):

    octaves = 7
    # octaves = 4 + 1
    # heightmap_scale = 16.0 * 4

    height_map = get_fractal_noise(width=width, height=height, octaves=octaves)
    normalize_array(height_map)

    return height_map




def test_erosion4(width=512, height=512, filename="erosion4"):
    erosion = cuda_texture_gen.Erosion4()


    

    heightmap_scale = 16.0 * 4

    erosion.mode = 1
    erosion.slope_threshold = 1.0
    # erosion.deposition_rate = 1.0


    height_map = get_heightmap_01(width, height)

    save_array_as_image(height_map * 255, "{}/{}.png".format(folder, filename + ".height"))
    height_map *= heightmap_scale
    erosion.height_map = height_map

    print(object_pars_to_dict(erosion))
    erosion.process()

    eroded_map = erosion.height_map
    print("eroded_map min: {}, max: {}".format(eroded_map.min(), eroded_map.max()))
    normalize_array(eroded_map)
    save_array_as_image(eroded_map * 255, "{}/{}.png".format(folder, filename + ".final_height"))

test_erosion4()

def test_erosion5(width=512, height=512, filename="erosion5"):

    erosion_preset_01()

 
    heightmap_scale = 16.0 * 4

    # ⛰️ height_map
    


    save_array_as_image(height_map * 255, "{}/{}.png".format(folder, filename + ".height"))
    height_map *= heightmap_scale
    erosion.height_map = height_map

    # 💧 start_water
    # water_map = get_fractal_noise(width=width, height=height, octaves=3, base_seed=2567)
    # normalize_array(water_map)
    # save_array_as_image(water_map * 255, "{}/{}.png".format(folder, filename + ".water"))
    # erosion.water_map = water_map

    # 🚀 launch process
    erosion.process()

    # ⛰️ height_map
    eroded_map = erosion.height_map
    print("eroded_map min: {}, max: {}".format(eroded_map.min(), eroded_map.max()))
    normalize_array(eroded_map)
    save_array_as_image(eroded_map * 255, "{}/{}.png".format(folder, filename + ".final_height"))

    # normal_map = generate_normal_map(array, strength=4.0)
    # save_array_as_image(normal_map * 255, "{}/{}.png".format(folder, base_filename + ".erode.normal"))
    # ao_map = generate_ao_map(array * 64, radius=3.0, mode=0)
    # save_array_as_image(ao_map * 255, "{}/{}.png".format(folder, base_filename + ".erode.ao"))
    # albedo = apply_color_map(array)
    # save_array_as_image(albedo * 255, "{}/{}.png".format(folder, base_filename + ".albedo"))

    water_map = erosion.water_map
    normalize_array(water_map)

    # blur(water_map, 0.5)
    # save_array_as_image(water_map * 255, "{}/{}.png".format(folder, filename + ".water.blur"))

    sediment_map = erosion.sediment_map
    save_array_as_image(sediment_map * 255, "{}/{}.png".format(folder, filename + ".final_sediment"))
    save_array_as_image(water_map * 255, "{}/{}.png".format(folder, filename + ".final_water"))

    # merged
    merged = merge_numpy_arrays_to_rgba(r=sediment_map, g=eroded_map / 2, b=water_map)
    save_array_as_image(merged * 255, "{}/{}.png".format(folder, filename + ".merged"))

    # blurs
    # blur_amount = 1.0
    # blur(eroded_map, blur_amount)
    # blur(water_map, blur_amount)
    # merged = merge_numpy_arrays_to_rgba(g=eroded_map, b=water_map)
    # save_array_as_image(merged * 255, "{}/{}.png".format(folder, filename + ".merged.blur"))


# test_erosion5()
