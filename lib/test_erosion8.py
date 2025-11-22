"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""


# from tools import *
import matplotlib
from matplotlib.colors import to_rgb
from scipy.ndimage import zoom
import tools
import cuda_texture_gen
import imageio.v2 as imageio  # v2 uses numpy arrays
import numpy as np

from typing import Dict, Any
import numpy.typing as npt

from collections import OrderedDict
from typing import OrderedDict as OrderedDictType

from tabulate import tabulate  # pip install tabulate
from collections import OrderedDict

import time


# --------print--------
def get_heightmap_01(width=128, height=128):

    octaves = 7
    # octaves = 4 + 1
    # heightmap_scale = 16.0 * 4

    height_map = tools.get_fractal_noise(width=width, height=height, octaves=octaves)
    tools.normalize_array(height_map)

    return height_map


class ErosionRunner:

    debug = True  # print debug information (note slows us down a bit)

    erosion = cuda_texture_gen.Erosion8()

    folder = "E:/"
    filename_base = "erosion"

    animation_fps = 5

    build_height_map_animation = True
    build_water_map_animation = True
    build_sediment_map_animation = True
    build_combined_map_animation = True

    frame_count = 64

    steps_per_frame = 8

    nearest_neighbor_upscale = 1  # if > 1 upscale the pixel sizes (makes it easier to see what's going on)

    process_time = 0.0

    def get_filename_start(self):
        return f"{self.folder}/{self.filename_base}"

    _default_pars: dict

    def __init__(self) -> None:
        self._default_pars = tools.object_pars_to_dict(self.erosion)

    def get_erosion_pars(self) -> Dict[str, Any]:
        pars = tools.dict_changes(
            self._default_pars,
            tools.object_pars_to_dict(self.erosion)
        )
        return pars

    def set_erosion_pars(self, pars: Dict[str, Any]) -> None:
        tools.set_object_with_dict(self.erosion, pars)

    def save_json(self) -> None:
        tools.save_dict_to_json(self.get_erosion_pars(), self.get_filename_start() + ".settings.json")

    def load_json(self):
        dict = tools.load_dict_from_json(f"{self.filename_base}.settings.json")
        tools.set_object_with_dict(self._erosion, dict)

    def PRESET_simple_erosion(self):

        # self.erosion.slope_threshold = 1.0  # optional will stop light slopes deteriorating
        self.erosion.simple_erosion_rate = 0.01
        # self.erosion.outflow_carve = 0.01

    def PRESET_erosion_01(self) -> None:
        """
        we have some working settings here
        """
        erosion = self.erosion

        # add water
        erosion.rain_rate = 0.05
        erosion.rain_rate = 0.01

        erosion.max_water_outflow = 1.0
        erosion.erosion_mode = 1
        erosion.erosion_mode = 1
        erosion.deposition_rate = 0.5 / 10.0
        erosion.erosion_rate = 0.1

        # remove water
        erosion.evaporation_rate = 0.001
        erosion.drain_at_min_height = True
        erosion.min_height = 0.0

        # trying to spread water
        # self._erosion.diffusion_rate = 0.001 # seems buggy
        # self._erosion.max_water_outflow = 0.2

        # changed_pars = dict_changes(default_pars, object_pars_to_dict(erosion))
        # print("🏔️", changed_pars)

    def PRESET_erosion_02(self) -> None:
        """
        we have some working settings here
        """
        erosion = self.erosion

        # add water
        erosion.rain_rate = 0.05
        erosion.rain_rate = 0.01

        erosion.max_water_outflow = 1.0
        erosion.erosion_mode = 1
        erosion.erosion_mode = 1
        erosion.deposition_rate = 0.5 / 10.0
        erosion.erosion_rate = 0.1

        # remove water
        erosion.evaporation_rate = 0.001
        erosion.drain_at_min_height = True
        erosion.min_height = 0.0

        # trying to spread water
        # self._erosion.diffusion_rate = 0.001 # seems buggy
        # self._erosion.max_water_outflow = 0.2

        # changed_pars = dict_changes(default_pars, object_pars_to_dict(erosion))
        # print("🏔️", changed_pars)

    erosion_map_names = ["height_map", "water_map", "sediment_map"]

    _maps: Dict[str, npt.NDArray[np.float32]] = {}

    def _download_maps(self):
        self._maps.clear()
        for name in self.erosion_map_names:
            self._maps[name] = getattr(self.erosion, name)

    def generate_meta_data(self) -> OrderedDictType:
        result: OrderedDictType = OrderedDict()

        for name in self.erosion_map_names:
            entry = OrderedDict()
            map = self._maps[name]
            entry['min'] = map.min()
            entry['max'] = map.max()
            entry['mean'] = map.mean()
            entry['std'] = map.std()
            result[name] = entry
        return result

    def print_meta_data(self) -> None:
        """
        print meta data table
        """
        meta_data = self.generate_meta_data()

        # Build rows
        rows = []
        for name, stats in meta_data.items():
            rows.append([name, stats['min'], stats['max'], stats['mean'], stats['std']])

        # Print as table
        print(tabulate(rows, headers=["Name", "Min", "Max", "Mean", "Std"], floatfmt=".3f"))

    class MapProfile:
        """
        🚧 UNUSED

        """

        name = "combined"

        # channels = ["height_map"]
        channels = ["height_map", "height_map", "water_map"]
        # channels = ["height_map", "height_map", "sediment_map"]

        clip = [None, None, None]  # if a number, clip at that number
        normalize = [True, False, False]

        def get_map(self, maps: Dict[str, npt.NDArray[np.float32]]):
            """
            get the final map with processing
            """
            maps = maps.copy()  # will shallow copy the maps to ensure we don't make changes to the orginal

            process_maps = {}

            for i in range(len(self.channels)):

                pass

    def process(self):

        start_time = time.perf_counter()
        erosion = self.erosion
        erosion.allocate_device()

        if self.debug:
            self._download_maps()  # ⚠️ donwnloads even though it uploaded (slow)
            print("🚀 launch erosion...")
            print("-" * 64)
            self.print_meta_data()  # ⚠️ gets meta data (slow)

        erosion.steps = self.steps_per_frame

        self.save_json()  # saves the erosion settings

        def get_mpg_writer(label=""):
            return imageio.get_writer(f"{self.folder}/{self.filename_base}{label}.mp4", fps=self.animation_fps)

        if self.build_height_map_animation:
            height_map_writer = get_mpg_writer()
        if self.build_water_map_animation:
            water_map_writer = get_mpg_writer(".water")
        if self.build_combined_map_animation:
            combined_map_writer = get_mpg_writer(".combined")

        for i in range(self.frame_count):

            self.erosion.process()

            self._download_maps()
            maps = self._maps.copy()  # shallow copy

            # print(f"frame: {i}, height_map min: {maps['height_map'].min():.2f}, max: {maps['height_map'].max():.2f}")

            # CLIP
            # maps['height_map'] = maps['height_map'].clip(0, 1)
            # maps['water_map'] = maps['water_map'].clip(0, 1)
            # maps['sediment_map'] = maps['sediment_map'].clip(0, 1)

            # NORMALIZE
            maps['height_map'] = tools.normalized_array(maps['height_map'])
            maps['water_map'] = tools.normalized_array(maps['water_map'])
            maps['sediment_map'] = tools.normalized_array(maps['sediment_map'])

            # nearest upscale (allows seeing the erosion)
            if self.nearest_neighbor_upscale > 1:
                for name, map in maps.items():
                    maps[name] = np.repeat(np.repeat(map, self.nearest_neighbor_upscale, axis=0), self.nearest_neighbor_upscale, axis=1)

            if self.build_height_map_animation:
                height_map_writer.append_data((maps['height_map'] * 255.0).astype(np.uint8))

            if self.build_water_map_animation:
                water_map_writer.append_data((maps['water_map'] * 255.0).astype(np.uint8))

            if self.build_combined_map_animation:
                # merged_array = tools.merge_numpy_arrays_to_color(maps['height_map'], maps['height_map'], maps['water_map'])
                merged_array = tools.merge_numpy_arrays_to_color(maps['sediment_map'], maps['height_map'], maps['water_map'])
                combined_map_writer.append_data((merged_array * 255.0).astype(np.uint8))

        end_time = time.perf_counter()
        self.process_time = end_time - start_time

        if self.debug:
            print("-" * 64)
            self.print_meta_data()
            print("-" * 64)
            print(f"process time: {self.process_time:.3f} seconds")

        # metadata = {
        #     "height_map.min": height_map.min(),
        #     "height_map.max": height_map.max(),
        #     "water_map.min": water_map.min(),
        #     "water_map.max": water_map.max(),
        #     "sediment_map.min": sediment_map.min(),
        #     "sediment_map.max": sediment_map.max(),
        # }

        # tools.save_array_as_image(height_map * 255, self.get_filename_start() + ".height.png")
        # tools.save_array_as_image(water_map * 255, self.get_filename_start() + ".water.png")
        # tools.save_array_as_image(sediment_map * 255, self.get_filename_start() + ".sediment.png")


# 2D array with shape (0, 0) .... USE FOR CLEARING A HEIGHTMAP.. OPTIONAL
empty_map = np.empty((0, 0), dtype=np.float32)


def test01():
    """
    shows okay rivers
    """
    height_map = tools.get_fractal_noise(width=256, height=256, octaves=5)
    tools.normalize_array(height_map)

    runner = ErosionRunner()
    runner.PRESET_erosion_01()
    # runner.PRESET_erosion_02()
    # erosion.PRESET_simple_erosion()

    runner.nearest_neighbor_upscale = 2
    runner.erosion.height_map = height_map * 128.0
    # runner.erosion.height_map = get_heightmap_01() * 32.0

    runner.process()

# test()


def lerp(a, b, t: float):
    return (1.0 - t) * a + t * b


def get_island_height_map(width, height):

    island = tools.load_image_as_array("./images/island03.png")
    island = tools.resize_array_2d(island, width, height)
    island = tools.blur_array_2d(island, 8)

    # tools.normalize_array(island)

    octaves = 8
    noise = tools.fractal_noise(width=width, height=height, octaves=octaves)
    tools.normalize_array(noise)

    fade = 0.5

    result = island * noise

    # result = lerp(result, noise, 0.7)

    tools.save_array_as_image(result, "./images/island03.out.png")

    return result


# get_island_height_map(512, 512)


def test02():
    """
    shows okay rivers
    """
    runner = ErosionRunner()
    erosion = runner.erosion

    height_map = empty_map
    water_map = empty_map
    sediment_map = empty_map
    rain_map = empty_map
    ################################################################
    map_width, map_height = 1024, 1024
    map_width, map_height = 512, 512

    # map_width, map_height = 256, 256
    # runner.nearest_neighbor_upscale = 2

    # map_width, map_height = 128, 128
    # runner.nearest_neighbor_upscale = 4
    ################################################################
    octaves = 8
    # octaves = 2
    # octaves = 4

    height_map = tools.get_fractal_noise(width=map_width, height=map_height, octaves=octaves)
    # height_map = tools.normalized_array(height_map)

    # height_map = get_island_height_map(map_width, map_height)

    # rain_map = height_map.copy() # rain map based on height
    # rain_map *= 128.0 #❗got a strange result with no copy
    # rain_map = rain_map / 2.0 + 0.5

    # rain_map = 1.0 - rain_map # invert
    height_map *= 128.0

    # add water
    erosion.rain_rate = 0.005

    erosion.max_water_outflow = 1.0
    erosion.erosion_mode = 1
    # erosion.erosion_mode = 2
    erosion.deposition_rate = 0.5 / 10.0
    # erosion.deposition_rate = 0.5
    # erosion.deposition_rate = 0.9
    erosion.erosion_rate = 0.1 / 4.0
    # erosion.erosion_rate = 0.2

    # remove water
    erosion.evaporation_rate = 0.001
    erosion.drain_at_min_height = True
    # erosion.min_height = 0.0

    # trying to spread water
    # self._erosion.diffusion_rate = 0.001 # seems buggy
    # self._erosion.max_water_outflow = 0.2

    # changed_pars = dict_changes(default_pars, object_pars_to_dict(erosion))
    # print("🏔️", changed_pars)

    ################################################################
    erosion.height_map = height_map
    erosion.water_map = water_map
    erosion.sediment_map = sediment_map
    erosion.rain_map = rain_map

    runner.steps_per_frame = 16
    runner.steps_per_frame = 64
    runner.process()


# test02()


def get_soil_pallete():

    soil_colors = np.array([
        to_rgb("#8b4513"),  # SaddleBrown (clay-rich soil)
        to_rgb("#f4a460"),  # SandyBrown (sand/loam)
        to_rgb("#d2b48c"),  # Tan (silt or light soil)
        to_rgb("#696969"),  # DimGray (shale/rock fragments)
        to_rgb("#deb887"),  # BurlyWood (weathered sandstone)
        to_rgb("#2f4f4f"),  # DarkSlateGray (basalt/organic-rich soil)
    ], dtype=np.float32)

    return soil_colors


def test_3d_arrays():

    soil_colors = np.array([
        to_rgb("#8b4513"),  # [139/255, 69/255, 19/255] → saddle brown
        to_rgb("#a0522d"),  # [160/255, 82/255, 45/255] → sienna
        to_rgb("#d2b48c"),  # [210/255, 180/255, 140/255] → tan
    ], dtype=np.float32)

    noise = tools.fractal_noise_rgb(256, 256).astype(np.float32)  # expect [0,1] floats

    den = noise.sum(axis=-1, keepdims=True)
    weights = noise / np.clip(den, 1e-8, None)

    noise_to_pallete = weights @ soil_colors  # matrix multiply (256, 256, 3)

    # island = cuda_texture_gen.blur

    tools.save_array_as_image((noise_to_pallete * 255).astype(np.uint8), "./output/noise.pallete.png")

    # noise = np.ascontiguousarray(noise[:, :, 0], dtype=np.float32)
    noise = noise[:, :, 0]
    print("🐈 ", type(noise))
    print("shape:", noise.shape)        # dimensions
    print("dtype:", noise.dtype)        # element type
    print("ndim:", noise.ndim)          # number of dimensions
    print("strides:", noise.strides)    # byte steps between elements
    print("flags:", noise.flags)        # contiguity info

    # noise = tools.fractal_noise(256, 256)s
    # print("🐈 ", type(noise))
    # print("shape:", noise.shape)        # dimensions
    # print("dtype:", noise.dtype)        # element type
    # print("ndim:", noise.ndim)          # number of dimensions
    # print("strides:", noise.strides)    # byte steps between elements
    # print("flags:", noise.flags)        # contiguity info

    cuda_texture_gen.blur(noise, 32)

    tools.save_array_as_image((noise * 255).astype(np.uint8), "./output/noise.rgb.png")


# test_3d_arrays()

def test_3d_arrays():

    # noise = tools.fractal_noise_rgb(256, 256).astype(np.float32)  # expect [0,1] floats
    noise = tools.fractal_noise(256, 256).astype(np.float32)  # expect [0,1] floats

    runner = ErosionRunner()
    erosion = runner.erosion

    erosion.layered_height_map = noise

    noise = erosion.layered_height_map

    tools.print_array_information(noise)


# test_3d_arrays()

def layers_test():
    runner = ErosionRunner()
    erosion = runner.erosion
    # print("layers_name:", erosion.layers_name)
    print("layers_resistance:", erosion.layers_resistance)
    print("layers_yield:", erosion.layers_yield)
    print("layers_permeability:", erosion.layers_permeability)
    print("layers_threshold:", erosion.layers_threshold)


# layers_test()





