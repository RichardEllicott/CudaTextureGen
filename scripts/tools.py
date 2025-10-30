"""

"""
import python_bootstrap  # bootstrap to our fresh compiled module
import cuda_texture_gen
import numpy as np
from PIL import Image
import inspect
from matplotlib.colors import to_rgb


def print_current_function():
    """
    print function call
    """
    frame = inspect.currentframe().f_back  # caller's frame
    func_name = frame.f_code.co_name
    args, _, _, values = inspect.getargvalues(frame)
    arg_str = ', '.join(f"{arg}={values[arg]!r}" for arg in args)
    print(f"{func_name}({arg_str})...")


def save_array_as_image(array, filename):
    """
    save numpy 2d array as an image (supports .png or .tif)
    """

    if isinstance(filename, (list, tuple)):
        for fname in filename:
            self.save_array_as_image(array, fname)
        return

    if filename.endswith(".png"):
        img = Image.fromarray(array.astype(np.uint8))
        img.save(filename)

    elif filename.endswith((".tif", ".tiff")):
        img = Image.fromarray(array.astype(np.float32))
        img.save(filename)

    else:
        raise ValueError(f"Unsupported format: {filename}")


def load_array_from_image(filename):
    """
    load image as black and white array
    """
    img = Image.open(filename).convert("L")
    array = np.array(img, dtype=np.float32)
    return array


def normalize_array(array):
    """
    normalize array in place (make from 0 to 1)
    """
    array -= array.min()
    array /= array.max()


def offset_array(array):
    """
    offset array by half (to test tiling)
    """

    # Compute half offsets
    dx = array.shape[1] // 2  # width
    dy = array.shape[0] // 2  # height

    # Apply toroidal (wraparound) shift
    # shifted = np.roll(arr, shift=(dy, dx), axis=(0, 1))
    array[:] = np.roll(array, shift=(dy, dx), axis=(0, 1))


def get_fractal_noise(width=1024, height=1024, octaves=6, base_period=2, base_seed=0, gain=0.8, lacunarity=2.0):
    """
    get_fractal_noise
    """
    gen = cuda_texture_gen.NoiseGenerator()
    gen.type = 0  # Assuming Perlin or similar

    array = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    period = base_period

    for i in range(octaves):
        gen.period = int(period)
        gen.seed = base_seed + i

        layer = gen.generate(width, height)
        normalize_array(layer)

        array += layer * amplitude
        total_amplitude += amplitude

        period *= lacunarity
        amplitude *= gain

    array /= total_amplitude  # Normalize final result
    return array


def erode_heightmap(height_map,
                    rain_rate=0.002,
                    max_water_outflow=0.005,
                    capacity=0.5,
                    erode=0.02,
                    deposit=0.005,
                    evaporation_rate=0.006,
                    steps=1024,
                    wrap=True

                    ):
    """
    run erosion
    """
    erosion = cuda_texture_gen.Erosion3()

    erosion.rain_rate = rain_rate
    erosion.max_water_outflow = max_water_outflow
    erosion.capacity = capacity
    erosion.erode = erode
    erosion.deposit = deposit
    erosion.evaporation_rate = evaporation_rate
    erosion.steps = steps
    erosion.wrap = wrap

    erosion.height_map = height_map
    erosion.process()
    height_map = erosion.height_map

    return height_map


def generate_normal_map(array, strength=1.0, wrap=True):
    """
    generate normal map
    """
    shader_maps = cuda_texture_gen.ShaderMaps()
    normal_map = shader_maps.generate_normal_map(array, strength, wrap)
    return normal_map


def generate_ao_map(array, strength=1.0, wrap=True):
    """
    generate ambient occlusion
    """
    shader_maps = cuda_texture_gen.ShaderMaps()
    ao_map = shader_maps.generate_ao_map(array)
    return ao_map


def smooth_layered_gradient(height_map, band_colors=[
        to_rgb("#8fb5f8"),
        to_rgb("#bed27b"),
        to_rgb("#f4a460"),
        to_rgb("#34783a"),
        to_rgb("#adcab1"),
    ]
):
    """
    height_map: 2D array normalized to [0, 1]
    band_colors: list of RGB tuples (e.g., from to_rgb)
    """
    bands = np.linspace(0, 1, len(band_colors))
    rgb_map = np.zeros((*height_map.shape, 3), dtype=np.float32)

    for i in range(len(bands) - 1):
        lower, upper = bands[i], bands[i + 1]
        mask = (height_map >= lower) & (height_map <= upper)
        t = (height_map[mask] - lower) / (upper - lower)  # blend factor [0,1]

        c0 = np.array(band_colors[i])
        c1 = np.array(band_colors[i + 1])
        rgb_map[mask] = (1 - t)[:, None] * c0 + t[:, None] * c1

    return rgb_map


def blur(input, amount = 1.0, wrap=True):
    cuda_texture_gen.blur(input, amount, wrap)
