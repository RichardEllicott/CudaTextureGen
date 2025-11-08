"""
testing 

"""

from tools import *
import numpy as np


def merge_numpy_arrays_to_rgba(r=None, g=None, b=None, a=None, shape=None, dtype=np.uint8):
    """
    Merge separate R, G, B (and optional A) numpy arrays into a single RGBA image.
    Missing channels default to black (0). Alpha defaults to fully opaque.

    Parameters
    ----------
    r, g, b, a : np.ndarray or None
        Arrays of the same shape, representing channels. Any can be None.
    shape : tuple or None
        Shape to use if some channels are None. If None, inferred from the first non-None channel.
    dtype : np.dtype
        Data type of the output array (default uint8).

    Returns
    -------
    rgba : np.ndarray
        Combined array of shape (H, W, 4).
    """
    # infer shape from first non-None channel
    if shape is None:
        for ch in (r, g, b, a):
            if ch is not None:
                shape = ch.shape
                dtype = ch.dtype
                break
    if shape is None:
        raise ValueError("Must provide at least one channel or a shape")

    def ensure_channel(ch, fill_value):
        if ch is None:
            return np.full(shape, fill_value, dtype=dtype)
        if ch.shape != shape:
            raise ValueError("Channel shape mismatch")
        return ch

    r = ensure_channel(r, 0)
    g = ensure_channel(g, 0)
    b = ensure_channel(b, 0)

    if a is None:
        # opaque: max for integers, 1.0 for floats
        if np.issubdtype(dtype, np.integer):
            fill = np.iinfo(dtype).max
        else:
            fill = 1.0
        a = np.full(shape, fill, dtype=dtype)
    else:
        a = ensure_channel(a, 0)

    return np.stack([r, g, b, a], axis=-1)


def test_erosion5(width=512, height=512, folder="output", filename="erosion5"):

    octaves = 7
    heightmap_scale = 16.0

    erosion = cuda_texture_gen.Erosion5()
    print("erosion:", dir(erosion))

    # erosion.debug_diagonal_distance = False
    erosion.steps = 1024
    # erosion.rain_rate = 0.01
    # erosion.rain_rate = 0.0
    # erosion.diffusion_rate = 0.003 * 0.0
    erosion.max_water_outflow = 1.0

    # erosion.slope_jitter = 1.0 / 10.0

    erosion.outflow_erode = 0.002
    # erosion.inflow_erode = 0.0005

    # erosion.min_height = 0.0


    # ⛰️ height_map
    height_map = get_fractal_noise(width=width, height=height, octaves=octaves)
    normalize_array(height_map)
    save_array_as_image(height_map * 255, "{}/{}.png".format(folder, filename + ".height"))
    height_map *= heightmap_scale
    erosion.height_map = height_map

    # 💧 start_water
    water_map = get_fractal_noise(width=width, height=height, octaves=3, base_seed=2567)
    normalize_array(water_map)
    save_array_as_image(water_map * 255, "{}/{}.png".format(folder, filename + ".water"))
    erosion.water_map = water_map

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
