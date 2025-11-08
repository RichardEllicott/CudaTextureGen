"""
testing 

"""

from tools import *
import numpy as np


def test_erosion5(width=512, height=512, folder="output", filename="erosion5"):

    # width *= 2
    # height *= 2

    octaves = 7
    heightmap_scale = 16.0

    erosion = cuda_texture_gen.Erosion5()
    print("erosion:", dir(erosion))

    # erosion.debug_diagonal_distance = False
    erosion.steps = 1024
    erosion.rain_rate = 0.05


    # erosion.diffusion_rate = 0.003
    erosion.max_water_outflow = 1.0

    erosion.erosion_mode = 1

    erosion.deposition_rate = 0.5

    # erosion.slope_jitter = 1.0 / 10.0

    erosion.erosion_rate = 0.002 * 3.0

    erosion.evaporation_rate = 0.03

    erosion.min_height = 0.0
    # erosion.min_height = -16.0


    # ⛰️ height_map
    height_map = get_fractal_noise(width=width, height=height, octaves=octaves)
    normalize_array(height_map)
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

    erosion.erosion_mode = 1

    # ⛰️ height_map
    eroded_map = erosion.height_map
    print("eroded_map min: {}, max: {}".format(eroded_map.min(), eroded_map.max()))
    # eroded_map = np.nan_to_num(eroded_map, nan=0.0, posinf=255.0, neginf=0.0)

    print("NaNs:", np.isnan(eroded_map).any())
    print("Infs:", np.isinf(eroded_map).any())




    normalize_array(eroded_map)


    print(type(eroded_map))

    print("eroded_map min: {}, max: {}".format(eroded_map.min(), eroded_map.max()))

    

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

    save_array_as_image(erosion.sediment_map * 255, "{}/{}.png".format(folder, filename + ".final_sediment"))
    save_array_as_image(water_map * 255, "{}/{}.png".format(folder, filename + ".final_water"))

    # merged
    merged = merge_numpy_arrays_to_rgba(g=eroded_map, b=water_map)
    save_array_as_image(merged * 255, "{}/{}.png".format(folder, filename + ".merged"))

    # blurs
    blur_amount = 1.0
    blur(eroded_map, blur_amount)
    blur(water_map, blur_amount)
    merged = merge_numpy_arrays_to_rgba(g=eroded_map, b=water_map)
    save_array_as_image(merged * 255, "{}/{}.png".format(folder, filename + ".merged.blur"))


test_erosion5()
