"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""

from tools import *

folder = "output"
folder = "E:/"
# folder = "godot/cuda_texture_gen/textures"
map_width, map_height = 512, 512
map_width, map_height = 256, 256
map_width, map_height = 128, 128
# map_width, map_height = 64, 64


erosion = cuda_texture_gen.Erosion7()
default_pars = object_pars_to_dict(erosion)


print(dir(cuda_texture_gen))

print(default_pars)


def default_pars_to_code_template() -> str:

    result = ""
    result += "def preset_example():\n"
    result += dict_to_string(default_pars, "    erosion.{key} = {value}\n")
    result += "    pass\n"
    return result

print(default_pars_to_code_template())



def save_erosion_settings(filename: str) -> None:
    pars = dict_changes(default_pars, object_pars_to_dict(erosion))
    save_dict_to_json(pars, filename)


def load_erosion_settings(filename) -> None:
    pars = dictionary_helpers.load_dict_from_json(filename)
    set_object_with_dict(erosion, pars)


def simple_erosion_preset() -> None:
    """
    same as my previous erosion model
    """
    # erosion.slope_threshold = 1.0 # optional will stop light slopes deteriorating
    erosion.simple_erosion_rate = 0.01

    # erosion.outflow_carve = 0.01


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

    # changed_pars = dict_changes(default_pars, object_pars_to_dict(erosion))
    # print("🏔️", changed_pars)

    return erosion


def erosion_preset_01_mod():

    # erosion.evaporation_rate = 0.06 * 2
    erosion.rain_rate /= 2.0
    # erosion.erosion_rate = 0.01

    erosion.debug_drain = True


    # erosion.diffusion_rate = 0.001 # seems buggy
    erosion.max_water_outflow = 0.2

    pass


def erosion_preset_02() -> None:
    # suggested
    erosion.rain_rate = 0.01  # 0.001–0.01
    erosion.evaporation_rate = 0.001
    erosion.diffusion_rate = 0.05  # or 0
    erosion.evaporation_rate = 0.05
    erosion.erosion_mode = 2
    erosion.erosion_rate = 0.02
    erosion.slope_exponent = 1.2  # to 1.5
    erosion.outflow_carve = 0.001  # to 0.01
    erosion.slope_threshold = 0.01  # to 0.05

    erosion.deposition_mode = 1
    erosion.deposition_rate = 0.001  # to 0.01
    erosion.sediment_capacity = 1.0  # to 2.0
    # erosion.steps = 2000 # to 4096
    erosion.slope_jitter = 0.01  # to 0.05

    # erosion.erosion_rate = 0.02
    # erosion.rain_rate = 0.01  # 0.001–0.01
    # erosion.evaporation_rate = 0.001
    # erosion.diffusion_rate = 0.0  # or 0
    # erosion.evaporation_rate = 0.05
    # erosion.erosion_mode = 2
    # erosion.slope_exponent = 1.2  # to 1.5
    # # erosion.outflow_carve = 0.001 # to 0.01
    # erosion.slope_threshold = 0.01  # to 0.05

    # # erosion.deposition_mode = 1
    # erosion.deposition_rate = 0.001  # to 0.01
    # # erosion.sediment_capacity = 1.0 # to 2.0
    # erosion.steps = 512  # to 4096
    # erosion.slope_jitter = 0.01  # to 0.05


def get_heightmap_01(width=512, height=512):

    octaves = 7
    # octaves = 4 + 1
    # heightmap_scale = 16.0 * 4

    height_map = get_fractal_noise(width=width, height=height, octaves=octaves)
    normalize_array(height_map)

    return height_map



"""
html example

<video src="erosion7_animation.mp4" autoplay loop muted></video>
"""




def test_erosion7_animation(filename_base="erosion7_animation"):

    # erosion_preset_01()
    # erosion_preset_01_mod()

    # erosion_preset_02()
    simple_erosion_preset()

    heightmap_scale = 16.0
    heightmap_scale = 1.0

    save_erosion_settings("{}/{}.json".format(folder, filename_base))

    height_map = get_heightmap_01(map_width, map_height)
    # save_array_as_image(height_map * 255, "{}/{}.png".format(folder, filename + ".start_height"))
    height_map *= heightmap_scale
    erosion.height_map = height_map

    frame_count = 64

    erosion.steps = 1024 // frame_count

    print(erosion.rain_rate)
    # erosion.rain_rate = 0.05

    animation_fps = 5

    # height_map_mpeg = imageio.get_writer(f"{folder}/{filename_base}.mp4", fps=animation_fps, codec='libx264')
    # water_map_mpeg = imageio.get_writer(f"{folder}/{filename_base}.water.mp4", fps=animation_fps, codec='libx264')

    height_map_mpeg = imageio.get_writer(f"{folder}/{filename_base}.mp4", fps=animation_fps)
    water_map_mpeg = imageio.get_writer(f"{folder}/{filename_base}.water.mp4", fps=animation_fps)
    combined_map_mpeg = imageio.get_writer(f"{folder}/{filename_base}.combined.mp4", fps=animation_fps)

    # height_map_mpeg.append_data((erosion.height_map / heightmap_scale * 255.0).astype(np.uint8))
    # water_map_mpeg.append_data((erosion.water_map / heightmap_scale * 1.0).astype(np.uint8))

    for i in range(frame_count):
        erosion.process()

        height_map = erosion.height_map
        water_map = erosion.water_map
        sediment_map = erosion.sediment_map

        print("height_map min: {}, max: {}".format(height_map.min(), height_map.max()))

        normalize_array(height_map)
        # normalize_array(water_map)

        # height_map.clip(0, 10, out=height_map)   # modifies arr directly
        water_map.clip(0, 8, out=water_map)   # modifies arr directly
        sediment_map.clip(0, 1, out=sediment_map)   # modifies arr directly

        # height_map /= heightmap_scale
        # water_map /= heightmap_scale

        height_map_mpeg.append_data((height_map * 255.0).astype(np.uint8))
        water_map_mpeg.append_data((water_map * 255.0).astype(np.uint8))

        # merged_array = merge_numpy_arrays_to_color(height_map, height_map, water_map)
        merged_array = merge_numpy_arrays_to_color(sediment_map, height_map, water_map)

        merged_array = nearest_neighbor_upscale(merged_array, 3)
        
        # merged_array = tile_array_2d(merged_array, 2, 1)
        
        combined_map_mpeg.append_data((merged_array * 255.0).astype(np.uint8))




    height_map_mpeg.close()
    water_map_mpeg.close()
    combined_map_mpeg.close()


test_erosion7_animation()
