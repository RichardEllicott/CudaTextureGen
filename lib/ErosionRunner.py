"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""

import tools
import cuda_texture_gen
from typing import Dict, Any
import numpy.typing as npt
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType
import numpy as np
from tabulate import tabulate  # pip install tabulate
import time
import imageio.v2 as imageio  # v2 uses numpy arrays
from numpy.typing import NDArray


from enum import Enum

# class Color(Enum):
#     RED   = 1
#     GREEN = 2
#     BLUE  = 3

# # Usage
# print(Color.RED)        # Color.RED
# print(Color.RED.value)  # 1
# print(Color.RED.name)   # "RED"


# from tools import *
# import matplotlib
# from matplotlib.colors import to_rgb
# from scipy.ndimage import zoom


# from collections import OrderedDict


class ErosionRunner:

    class Mode(Enum):
        normal = 0
        layer = 1

    mode = Mode.normal

    debug = True  # print debug information (note slows us down a bit)

    folder = "E:/"
    # folder = "./output/"
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

    # PROPERTIES

    # height_map = property(lambda self: self.erosion.height_map,
    #                 lambda self, v: setattr(self.erosion, "height_map", v))
    # layer_map = property(lambda self: self.layer_map.depth,
    #                  lambda self, v: self.erosion.)

    @property
    def layer_map(self):
        v = self.erosion.layer_map
        # stored internally as (C,H,W), return to Python as (H,W,C)
        return np.transpose(v, (1, 2, 0))

    @layer_map.setter
    def layer_map(self, v):
        # self.mode = self.Mode.layer  # layer mode
        # accept (H,W,C), store internally as (C,H,W)
        v = np.ascontiguousarray(np.transpose(v, (2, 0, 1)))
        self.erosion.layer_map = v
    # PROPERTIES

    _default_pars: dict

    def __init__(self) -> None:
        self.erosion: cuda_texture_gen.Erosion9 = cuda_texture_gen.Erosion9()  # or None if lazy init
        self._default_pars = tools.object_pars_to_dict(self.erosion)

    def get_erosion_pars(self) -> dict[str, Any]:
        pars = tools.dict_changes(
            self._default_pars,
            tools.object_pars_to_dict(self.erosion)
        )
        return pars

    def set_erosion_pars(self, pars: dict[str, Any]) -> None:
        tools.set_object_with_dict(self.erosion, pars)

    def save_json(self) -> None:
        tools.save_dict_to_json(self.get_erosion_pars(), self.filename_base + ".settings.json")

    def load_json(self) -> None:
        dict = tools.load_dict_from_json(f"{self.filename_base}.settings.json")
        tools.set_object_with_dict(self.erosion, dict)

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
            # print(f"download map: {name}")
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

        def get_map(self, maps: dict[str, NDArray[np.float32]]):
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
                merged_array = tools.merge_numpy_arrays_to_color(maps['height_map'], maps['height_map'], maps['water_map'])
                # merged_array = tools.merge_numpy_arrays_to_color(maps['sediment_map'], maps['height_map'], maps['water_map'])
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


def main():

    pass


if __name__ == "__main__":
    print("Running main logic...")
    main()
