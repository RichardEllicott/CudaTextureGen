"""
testing 

"""

from tools import *

folder = "output"
# folder = "godot/cuda_texture_gen/textures"


erosion = cuda_texture_gen.Erosion5()
default_pars = object_pars_to_dict(erosion)


print(dir(cuda_texture_gen))


def save_erosion_settings(filename) -> None:
    pars = dict_changes(default_pars, object_pars_to_dict(erosion))
    save_dict_to_json(pars, filename)


def load_erosion_settings(filename) -> None:
    pars = dictionary_helpers.load_dict_from_json(filename)
    set_object_with_dict(erosion, pars)


def simple_erosion_preset() -> None:
    """
    same as my previous erosion model
    """
    erosion.slope_threshold = 1.0
    # erosion.outflow_carve = 0.01
    erosion.simple_erosion_rate = 0.01


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




def erosion_preset_02() -> None:
    # suggested
    # erosion.rain_rate = 0.01 # 0.001–0.01
    # erosion.evaporation_rate = 0.001
    # erosion.diffusion_rate = 0.05 # or 0
    # erosion.evaporation_rate = 0.05
    # erosion.erosion_mode = 2
    # erosion.erosion_rate = 0.02
    # erosion.slope_exponent = 1.2 # to 1.5
    # erosion.outflow_carve = 0.001 # to 0.01
    # erosion.slope_threshold = 0.01 # to 0.05

    # erosion.deposition_mode = 1
    # erosion.deposition_rate = 0.001 # to 0.01
    # erosion.sediment_capacity = 1.0 # to 2.0
    # # erosion.steps = 2000 # to 4096 
    # erosion.slope_jitter = 0.01 # to 0.05


    erosion.erosion_rate = 0.02
    erosion.rain_rate = 0.01 # 0.001–0.01
    erosion.evaporation_rate = 0.001
    erosion.diffusion_rate = 0.0 # or 0
    erosion.evaporation_rate = 0.05
    erosion.erosion_mode = 2
    erosion.slope_exponent = 1.2 # to 1.5
    # erosion.outflow_carve = 0.001 # to 0.01
    erosion.slope_threshold = 0.01 # to 0.05

    # erosion.deposition_mode = 1
    erosion.deposition_rate = 0.001 # to 0.01
    # erosion.sediment_capacity = 1.0 # to 2.0
    erosion.steps = 512 # to 4096 
    erosion.slope_jitter = 0.01 # to 0.05


    

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
    erosion.erosion_rate = 0.01
    erosion.slope_threshold = 1.0

    height_map = get_heightmap_01(width, height)

    save_array_as_image(height_map * 255, "{}/{}.png".format(folder, filename + ".start_height"))
    height_map *= heightmap_scale
    erosion.height_map = height_map

    print(object_pars_to_dict(erosion))

    erosion.process()
    height_map = erosion.height_map
    print("height_map min: {}, max: {}".format(height_map.min(), height_map.max()))
    normalize_array(height_map)
    save_array_as_image(height_map * 255, "{}/{}.png".format(folder, filename + ".height"))

    sediment_map = erosion.sediment_map
    save_array_as_image(sediment_map * 255, "{}/{}.png".format(folder, filename + ".sediment"))


def test_erosion5(map_width=512, map_height=512, filename="erosion_preset_01"):

    erosion_preset_01()
    # erosion_preset_02()
    # simple_erosion_preset()

    heightmap_scale = 16.0

    save_erosion_settings("{}/{}.json".format(folder, filename))

    height_map = get_heightmap_01(map_width, map_height)
    save_array_as_image(height_map * 255, "{}/{}.png".format(folder, filename + ".start_height"))
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
    height_map = erosion.height_map
    print("height_map min: {}, max: {}".format(height_map.min(), height_map.max()))
    normalize_array(height_map)
    save_array_as_image(height_map * 255, "{}/{}.png".format(folder, filename + ".height"))

    water_map = erosion.water_map
    normalize_array(water_map)
    save_array_as_image(water_map * 255, "{}/{}.png".format(folder, filename + ".water"))

    sediment_map = erosion.sediment_map
    save_array_as_image(sediment_map * 255, "{}/{}.png".format(folder, filename + ".sediment"))

    normal_map = generate_normal_map(height_map)
    save_array_as_image(normal_map * 255, "{}/{}.png".format(folder, filename + ".normal"))

    # ao_map = generate_ao_map(array * 64, radius=3.0, mode=0)
    ao_map = generate_ao_map(height_map * 64)
    save_array_as_image(ao_map * 255, "{}/{}.png".format(folder, filename + ".ao"))

    albedo_map = apply_color_map(height_map)
    save_array_as_image(albedo_map * 255, "{}/{}.png".format(folder, filename + ".albedo"))

    # # merged
    # merged = merge_numpy_arrays_to_rgba(r=sediment_map, g=eroded_map / 2, b=water_map)
    # save_array_as_image(merged * 255, "{}/{}.png".format(folder, filename + ".merged"))

    # # blurs
    # # blur_amount = 1.0
    # # blur(eroded_map, blur_amount)
    # # blur(water_map, blur_amount)
    # # merged = merge_numpy_arrays_to_rgba(g=eroded_map, b=water_map)
    # # save_array_as_image(merged * 255, "{}/{}.png".format(folder, filename + ".merged.blur"))


test_erosion5()
