"""

ErosionRunner object, runs the erosion and outputs movie and/or images from the data

"""

import tools
import cuda_texture_gen
from typing import Any
import numpy.typing as npt
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType
import numpy as np
from tabulate import tabulate  # pip install tabulate
import time
import imageio.v2 as imageio  # v2 uses numpy arrays
from numpy.typing import NDArray
from enum import Enum
import json


class FrameProfile:
    """
    a profile of a frame, builds a frame from the channels list
    """

    # channels, list of strings or a string (for one channel)
    channels: list[str] = ["height_map", "height_map", "water_map"]

    clip: list[int | float | None] = [None, None, None]  # if a number, clip at that number
    normalize: list[bool] = [True, True, True]

    # generate as ao map
    ao_map: bool = False
    ao_map_strength: float = 1.0
    ao_map_radius: float = 1.0

    # generate as normal map
    normal_map: bool = False
    normal_map_strength: float = 1.0

    def validate(self) -> None:
        # Only allow 1 or 3 channels

        if len(self.channels) not in (1, 3):
            raise ValueError(
                f"FrameProfile must have 1 or 3 channels, got {len(self.channels)}: {self.channels}"
            )

        if self.normal_map and len(self.channels) != 1:
            raise ValueError(
                f"FrameProfile normal map must be 1 channel, got {len(self.channels)}: {self.channels}"
            )

    def __init__(self, runner: "ErosionRunner") -> None:
        self.runner = runner

    def get_frame(self) -> NDArray:
        """
        build and get a frame
        """

        self.validate()  # ensure 1 or 3 channels

        processed_maps = []

        channels = self.channels
        if isinstance(channels, str):
            channels = [channels]

        for i in range(len(channels)):
            channel = channels[i]

            if channel:  # if channel not None
                map = self.runner._maps[channel].copy()

                # clip
                clip = self.clip[i]
                if clip:
                    map = map.clip(0, clip)

                # normalize
                normalize: bool = self.normalize[i]
                if normalize:
                    tools.arrays.normalize(map)

                if self.ao_map:
                    if self.ao_map_strength != 0.0:
                        map *= self.ao_map_strength
                    map = tools.cuda.ao_map(map, self.ao_map_radius, True)

                if self.normal_map:
                    map = tools.cuda.normal_map(map, self.normal_map_strength, True)

            processed_maps.append(map)  # appends map or None

        if len(channels) == 1:
            return processed_maps[0]
        # elif len(self.channels) == 3:
        return tools.arrays.merge_to_color(*processed_maps)  # should be 3 channels


class MovieProfile(FrameProfile):

    # # Create an in-memory buffer
    # buf = io.BytesIO()

    fps: int = 30

    def __init__(self, runner: "ErosionRunner") -> None:
        super().__init__(runner)

        # self.move_writer = imageio.get_writer(filename, fps=fps)

        # self.writer = imageio.get_writer(self.buf, codec="libx264", fps=30)


class ErosionRunner:

    debug: bool = True  # print debug information (note slows us down a bit)

    class Mode(Enum):
        normal = 0
        layer = 1
    mode: Mode = Mode.normal

    folder: str = "E:/"
    # folder = "./output/"
    filename_base: str = "erosion"

    animation_fps: int = 5
    frame_count: int = 64
    steps_per_frame: int = 16
    nearest_neighbor_upscale: int = 1  # 🐞 if > 1 upscale the pixel sizes (makes it easier to see what's going on)
    process_time: float = 0.0

    # profiles build movies each frame in process
    movie_profiles: dict[str, MovieProfile] = {}
    # profiles build image at end of process
    image_profiles: dict[str, FrameProfile] = {}

    # names of the maps to download each frame
    map_names = ["height_map", "water_map", "sediment_map"]

    _maps: dict[str, npt.NDArray[np.float32]] = {}
    _default_pars: dict

    # PROPERTIES

    # height_map = property(lambda self: self.erosion.height_map,
    #                 lambda self, v: setattr(self.erosion, "height_map", v))
    # layer_map = property(lambda self: self.layer_map.depth,
    #                  lambda self, v: self.erosion.)

    # for changing to the internal format
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

    def __init__(self) -> None:
        self.erosion: cuda_texture_gen.Erosion9 = cuda_texture_gen.Erosion9()  # or None if lazy init
        self._default_pars = tools.dicts.from_object(self.erosion)
        self.output_preset_01()  # defaults

    def output_preset_01(self):

        self.movie_profiles = {}
        self.image_profiles = {}

        # combined map movie
        movie_profile = MovieProfile(self)
        movie_profile.channels = ["sediment_map", "height_map", "water_map"]  # all channels
        # movie_profile.channels = ["height_map", "height_map", "water_map"]  # height does yellow
        # movie_profile.clip = [None, None, 1.0]
        self.movie_profiles["combined"] = movie_profile

        # height map movie
        movie_profile = MovieProfile(self)
        movie_profile.channels = ["height_map"]
        self.movie_profiles["height"] = movie_profile

        # water map movie
        movie_profile = MovieProfile(self)
        movie_profile.channels = ["water_map"]
        self.movie_profiles["water"] = movie_profile

        # height map image
        image_profile = FrameProfile(self)
        image_profile.channels = ["height_map"]
        self.image_profiles["height.png"] = image_profile

        # normal map image
        image_profile = FrameProfile(self)
        image_profile.normal_map = True
        image_profile.normal_map_strength = 1.0
        image_profile.channels = ["height_map"]
        self.image_profiles["normal.png"] = image_profile

        # ao map image
        image_profile = FrameProfile(self)
        image_profile.ao_map = True
        image_profile.ao_map_strength = 16.0
        image_profile.channels = ["height_map"]
        self.image_profiles["ao.png"] = image_profile

    def output_preset_02(self):

        self.output_preset_01()
        # combined map movie
        movie_profile = MovieProfile(self)
        # movie_profile.channels = ["sediment_map", "height_map", "water_map"]  # all channels
        movie_profile.channels = ["height_map", "height_map", "water_map"]  # height does yellow
        # movie_profile.clip = [None, None, 1.0]
        self.movie_profiles["combined"] = movie_profile

    def output_preset_03(self):

        self.output_preset_01()
        # combined map movie
        movie_profile = MovieProfile(self)
        # movie_profile.channels = ["sediment_map", "height_map", "water_map"]  # all channels
        movie_profile.channels = ["height_map", "height_map", "water_map"]  # height does yellow
        movie_profile.clip = [None, None, 1.0]
        self.movie_profiles["combined"] = movie_profile

    def get_erosion_pars(self) -> dict[str, Any]:
        """
        get erosion as dict
        """
        pars = tools.dicts.changes(
            self._default_pars,
            tools.dicts.from_object(self.erosion)
        )
        return pars

    def set_erosion_pars(self, pars: dict[str, Any]) -> None:
        """
        set erosion pars from dict
        """
        tools.dicts.set_object(self.erosion, pars)

    def get_debug_data(self):

        debug = {}
        self.erosion.debug_update()
        tile_count = self.erosion._width * self.erosion._height

        for attr in dir(self.erosion):
            if attr.startswith("_debug"):
                value = getattr(self.erosion, attr)

                if isinstance(value, float):  # round floats
                    value /= float(tile_count)
                    debug[attr] = value

        return debug

    def save_meta_data(self):

        filename = f"{self.folder}{self.filename_base}.data.json"

        data = {}

        data["process_time"] = self.process_time

        data["erosion_settings"] = self.get_erosion_pars()

        runner_settings = {}
        runner_settings["frame_count"] = self.frame_count
        runner_settings["steps_per_frame"] = self.steps_per_frame
        data["runner_settings"] = runner_settings

        data["metrics"] = self.get_metric_data()

        if self.debug:
            data["debug"] = self.get_debug_data()

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def get_last_meta_data(self):
        filename = f"{self.folder}{self.filename_base}.data.json"
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

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

    def _download_maps(self):
        self._maps.clear()
        for name in self.map_names:
            map = getattr(self.erosion, name)
            self._maps[name] = map

    _metric_data = {}

    def get_metric_data(self) -> dict:

        if self._metric_data:
            return self._metric_data

        for name in self.map_names:
            entry: dict = {}
            map = self._maps[name]
            entry['min'] = float(map.min())
            entry['max'] = float(map.max())
            entry['mean'] = float(map.mean())
            entry['std'] = float(map.std())
            self._metric_data[name] = entry

        return self._metric_data

    def print_metric_data(self) -> None:
        """
        print meta data table
        """
        meta_data = self.get_metric_data()

        # Build rows
        rows = []
        for name, stats in meta_data.items():
            rows.append([name, stats['min'], stats['max'], stats['mean'], stats['std']])

        # Print as table
        print(tabulate(rows, headers=["Name", "Min", "Max", "Mean", "Std"], floatfmt=".4f"))

    def process(self):

        self._metric_data = {}

        start_time = time.perf_counter()
        erosion = self.erosion

        erosion._debug = self.debug
        if self.debug:
            erosion.allocate_device()  # pre-allocate to allow debug download
            self._download_maps()  # ⚠️ donwnloads even though it uploaded (slow)
            print("🚀 launch erosion...")
            print("-" * 64)
            self.print_metric_data()  # ⚠️ gets meta data (slow)

        erosion.steps = self.steps_per_frame

        movie_writers = {}
        for key in self.movie_profiles:
            writer = imageio.get_writer(f"{self.folder}/{self.filename_base}.{key}.mp4", fps=self.animation_fps)
            movie_writers[key] = writer

        for i in range(self.frame_count):

            self.erosion.process()

            self._download_maps()

            # 🐞 nearest upscale (allows seeing the erosion)
            if self.nearest_neighbor_upscale > 1:
                for name, map in self._maps.items():
                    self._maps[name] = np.repeat(np.repeat(map, self.nearest_neighbor_upscale, axis=0), self.nearest_neighbor_upscale, axis=1)

            # for each movie writer write using frame profile
            for key in movie_writers:
                profile = self.movie_profiles[key]
                frame = profile.get_frame()
                writer = movie_writers[key]
                writer.append_data((frame * 255.0).astype(np.uint8))

        end_time = time.perf_counter()
        self.process_time = end_time - start_time

        if self.debug:
            print("-" * 64)
            self.print_metric_data()
            print("-" * 64)

        print(f"process time: {self.process_time:.3f} seconds")

        for name in self.image_profiles:
            profile = self.image_profiles[name]
            frame = profile.get_frame()
            tools.images.save(frame, f"{self.folder}/{self.filename_base}.{name}")

        if self.debug:
            self.erosion.debug_update()
            tile_count = erosion._width * erosion._height

            for attr in dir(self.erosion):
                if attr.startswith("_debug"):
                    value = getattr(self.erosion, attr)

                    if isinstance(value, float):  # round floats
                        value /= float(tile_count)
                        print(f"{attr}: {value:.4f}")

        self.save_meta_data()


def main():
    pass


if __name__ == "__main__":
    print("Running main logic...")
    main()
