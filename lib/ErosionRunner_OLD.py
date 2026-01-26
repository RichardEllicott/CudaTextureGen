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
import warnings


class FrameProfile:
    """
    a profile of a frame, builds a frame from the channels list
    """

    # channels, list of strings or a string (for one channel)
    channels: list[str] = ["height_map", "height_map", "water_map"]

    clip: list[int | float | None] = [None, None, None]  # if a number, clip at that number
    normalize: list[bool] = [True, True, True]

    # class Type(Enum):
    #     default = 0
    #     normal_map = 1
    #     ao_map = 2
    #     layer_map = 3
    # type: Type = Type.default

    # generate as ao map
    ao_map: bool = False
    ao_map_strength: float = 1.0
    ao_map_radius: float = 1.0

    # generate as normal map
    normal_map: bool = False
    normal_map_strength: float = 1.0

    albedo_map_0: bool = False

    _gradient_strip = None

    def apply_gradient(self, map):

        if not self._gradient_strip:
            gradient: tools.gradients.Gradient = tools.gradients.get_test_gradient_02(0, 8)
            # gradient: tools.gradients.Gradient = tools.gradients.get_test_gradient_02(0, 128)
            self._gradient_strip = gradient.render(512)

        starting_heightmap = self.runner.starting_heightmap

        if starting_heightmap is not None and starting_heightmap.size > 0:
            map = map - starting_heightmap
        else:
            warnings.warn("starting_heightmap empty or missing!")

        # map = self.runner.starting_heightmap - map
        tools.arrays.normalize(map)

        map = tools.palettes.apply_gradient_strip(map, self._gradient_strip)
        return map

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

        if self.albedo_map_0 and len(self.channels) != 1:
            raise ValueError(
                f"FrameProfile albedo_map_0 must be 1 channel, got {len(self.channels)}: {self.channels}"
            )

    def __init__(self, runner: "Runner") -> None:
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

                # tools.arrays.print_array_information(map) # DEBUGING

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

                if self.albedo_map_0:
                    map = self.apply_gradient(map)

            processed_maps.append(map)  # appends map or None

        if len(channels) == 1:
            return processed_maps[0]
        # elif len(self.channels) == 3:
        return tools.arrays.merge_to_color(*processed_maps)  # should be 3 channels


class MovieProfile(FrameProfile):

    # # Create an in-memory buffer
    # buf = io.BytesIO()

    fps: int = 30

    def __init__(self, runner: "Runner") -> None:
        super().__init__(runner)

        # self.move_writer = imageio.get_writer(filename, fps=fps)

        # self.writer = imageio.get_writer(self.buf, codec="libx264", fps=30)


class Runner:

    # erosion: cuda_texture_gen.Erosion10  # the main erosion object

    _erosion_default_pars: dict[str, Any]  # filled with the starting pars of erosion

    folder: str = "E:/"  # output folder for movies and images
    # folder: str = "./output/"
    # folder: str = "./godot/cuda_texture_gen/projects/erosion_test/"

    filename_base: str = "erosion"  # base string for files

    debug: bool = True  # print debug information (note slows us down a bit)

    animation_fps: int = 5
    frame_count: int = 64
    steps_per_frame: int = 16
    nearest_neighbor_upscale: int = 1  # 🐞 if > 1 upscale the pixel sizes (makes it easier to see what's going on)
    process_time: float = 0.0

    # profiles build movies each frame in process
    movie_profiles: dict[str, MovieProfile] = {}
    # profiles build image at end of process
    image_profiles: dict[str, FrameProfile] = {}

    _maps: dict[str, npt.NDArray[np.float32]] = {}  # downloaded maps cache

    movie_profile_map_names: set[str] = set()  # contains all the map names to download each frame
    image_profile_maps_names: set[str] = set()  # contains the map names to download for the final images

    starting_heightmap: NDArray[np.float32] | None = None

    def __init__(self) -> None:

        self.erosion = cuda_texture_gen.Erosion10()
        self._erosion_default_pars = tools.dicts.from_object(self.erosion)

        self.OUTPUT_PRESET_01()  # defaults

        pass

    def OUTPUT_PRESET_01(self):

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

        # # height map image (tif)
        # image_profile = FrameProfile(self)
        # image_profile.channels = ["height_map"]
        # self.image_profiles["height.tif"] = image_profile

    #    # height map image (exr)
    #     image_profile = FrameProfile(self)
    #     image_profile.channels = ["height_map"]
    #     self.image_profiles["height.exr"] = image_profile

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

        # gradient albedo map
        image_profile = FrameProfile(self)
        image_profile.albedo_map_0 = True
        image_profile.channels = ["height_map"]
        self.image_profiles["albedo.png"] = image_profile

    def OUTPUT_PRESET_02(self):

        self.OUTPUT_PRESET_01()
        # combined map movie
        movie_profile = MovieProfile(self)
        # movie_profile.channels = ["sediment_map", "height_map", "water_map"]  # all channels
        movie_profile.channels = ["height_map", "height_map", "water_map"]  # height does yellow
        # movie_profile.clip = [None, None, 1.0]
        self.movie_profiles["combined"] = movie_profile

    def OUTPUT_PRESET_03(self):

        self.OUTPUT_PRESET_01()
        # combined map movie
        movie_profile = MovieProfile(self)
        # movie_profile.channels = ["sediment_map", "height_map", "water_map"]  # all channels
        movie_profile.channels = ["height_map", "height_map", "water_map"]  # height does yellow
        movie_profile.clip = [None, None, 1.0]
        self.movie_profiles["combined"] = movie_profile

    def OUTPUT_PRESET_add_layers_01(self) -> None:
        """
        setup ready for layer mode

        """
        # height map image
        image_profile = FrameProfile(self)
        image_profile.channels = ["layer_map"]
        image_profile.normalize = [True]
        self.image_profiles["layers.png"] = image_profile

    def get_erosion_pars(self) -> dict[str, Any]:
        """
        get erosion pars as dict, only changed pars
        """
        pars = tools.dicts.changes(
            self._erosion_default_pars,
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

    _metric_data = {}

    # # names of the maps to get metric data from
    metric_data_map_names: list[str] = ["height_map", "water_map", "sediment_map"]

    def get_metric_data(self) -> dict:

        if self._metric_data:
            return self._metric_data

        for name in self.metric_data_map_names:

            if not name in self._maps:
                self._maps[name] = getattr(self.erosion, name)

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

    def clear_downloaded_maps(self):
        self._maps.clear()

    def download_image_profile_maps(self):
        """
        download all image profile maps, if not already downloaded
        (need to clear if we expect the maps to have changed)
        """
        for map_name in self.image_profile_maps_names:
            if not map_name in self._maps:
                map = getattr(self.erosion, map_name)
                self._maps[map_name] = map

    def download_movie_profile_maps(self):
        """
        download all movie profile maps, if not already downloaded
        (need to clear if we expect the maps to have changed)
        """
        for map_name in self.movie_profile_map_names:
            if not map_name in self._maps:
                map = getattr(self.erosion, map_name)
                self._maps[map_name] = map

    def _setup(self):
        """
        run at start of process
        """
        # find all unique map names for the movie profiles (to download each frame)
        self.movie_profile_map_names: set[str] = set()
        for key in self.movie_profiles:
            movie_profile: MovieProfile = self.movie_profiles[key]
            for name in movie_profile.channels:
                self.movie_profile_map_names.add(name)

        # find all unique map names for the image profiles (to download at end of complete process)
        self.image_profile_maps_names: set[str] = set()
        for key in self.image_profiles:
            image_profile: FrameProfile = self.image_profiles[key]
            for name in image_profile.channels:
                self.image_profile_maps_names.add(name)

    def _compute(self):
        """
        hook to override
        """
        self.erosion.process()

        pass

    def process(self) -> None:
        """

        """
        self._setup()

        start_time = time.perf_counter()
        erosion = self.erosion

        self.starting_heightmap = erosion.height_map
        if self.starting_heightmap is not None and self.starting_heightmap.size > 0:

            tools.arrays.print_array_information(self.starting_heightmap)

            tools.images.save(
                tools.arrays.normalized(self.starting_heightmap),
                f"{self.folder}/{self.filename_base}.start.png")

        erosion._debug = self.debug
        if self.debug:
            erosion.allocate_device()  # pre-allocate to allow debug download
            # self._download_maps()  # ⚠️ downloads even though it uploaded (slow)
            print("🚀 launch erosion...")
            print("-" * 64)
            self.print_metric_data()  # ⚠️ gets meta data (slow)

        self._metric_data = {}

        erosion.steps = self.steps_per_frame

        movie_writers = {}
        for key in self.movie_profiles:
            writer = imageio.get_writer(f"{self.folder}/{self.filename_base}.{key}.mp4", fps=self.animation_fps)
            movie_writers[key] = writer

        for i in range(self.frame_count):

            self._compute()

            self.clear_downloaded_maps()  # ensures new maps are downloaded
            self.download_movie_profile_maps()

            # 🐞 nearest upscale (allows seeing the erosion)
            if self.nearest_neighbor_upscale > 1:
                for name, map in self._maps.items():
                    # self._maps[name] = np.repeat(np.repeat(map, self.nearest_neighbor_upscale, axis=0), self.nearest_neighbor_upscale, axis=1)
                    self._maps[name] = tools.arrays.nearest_neighbor_upscale(map, self.nearest_neighbor_upscale)

                # if self.starting_heightmap is not None:
                #     self.starting_heightmap = tools.arrays.nearest_neighbor_upscale(self.starting_heightmap, self.nearest_neighbor_upscale)

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

        self.clear_downloaded_maps()  # only required to clear the upscale i think
        self.download_image_profile_maps()

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


class ErosionRunner(Runner):
    """
    Default runner still working
    """

    pass


class ErosionRunnerGNC(Runner):

    def __init__(self) -> None:

        # self.erosion = cuda_texture_gen.Erosion10()

        self.erosion = cuda_texture_gen.GNC_Erosion()
        # self.erosion = cuda_texture_gen.GNC_Noise()

        self._erosion_default_pars = tools.dicts.from_object(self.erosion)

        self.OUTPUT_PRESET_01()  # defaults

    def download_image_profile_maps(self):
        """
        download all image profile maps, if not already downloaded
        (need to clear if we expect the maps to have changed)
        """
        for map_name in self.image_profile_maps_names:
            if not map_name in self._maps:
                map = getattr(self.erosion, map_name)
                self._maps[map_name] = map.array

    def download_movie_profile_maps(self):
        """
        download all movie profile maps, if not already downloaded
        (need to clear if we expect the maps to have changed)
        """
        for map_name in self.movie_profile_map_names:
            if not map_name in self._maps:
                map: cuda_texture_gen.DeviceArrayFloat2D = getattr(self.erosion, map_name)
                print(map_name, map)
                print(map_name, map.shape)
                self._maps[map_name] = map.array

    def _compute(self):
        """
        hook to override
        """
        self.erosion.compute()

    def process(self) -> None:
        """

        """
        self._setup()

        start_time = time.perf_counter()
        erosion = self.erosion

        self.starting_heightmap = erosion.height_map.array
        if self.starting_heightmap is not None and self.starting_heightmap.size > 0:

            tools.arrays.print_array_information(self.starting_heightmap)

            tools.images.save(
                tools.arrays.normalized(self.starting_heightmap),
                f"{self.folder}/{self.filename_base}.start.png")

        # erosion._debug = self.debug
        # if self.debug:
        #     erosion.allocate_device()  # pre-allocate to allow debug download
        #     # self._download_maps()  # ⚠️ downloads even though it uploaded (slow)
        #     print("🚀 launch erosion...")
        #     print("-" * 64)
        #     self.print_metric_data()  # ⚠️ gets meta data (slow)

        self._metric_data = {}

        erosion.steps = self.steps_per_frame

        movie_writers = {}
        for key in self.movie_profiles:
            writer = imageio.get_writer(f"{self.folder}/{self.filename_base}.{key}.mp4", fps=self.animation_fps)
            movie_writers[key] = writer

        for i in range(self.frame_count):

            self._compute()

            print("compute done!")
            # stream: cuda_texture_gen.

            # stream: cuda_texture_gen.Stream = self.erosion.stream
            # stream.sync()

            # 


            self.clear_downloaded_maps()  # ensures new maps are downloaded
            self.download_movie_profile_maps()

            # 🐞 nearest upscale (allows seeing the erosion)
            if self.nearest_neighbor_upscale > 1:
                for name, map in self._maps.items():
                    # self._maps[name] = np.repeat(np.repeat(map, self.nearest_neighbor_upscale, axis=0), self.nearest_neighbor_upscale, axis=1)
                    self._maps[name] = tools.arrays.nearest_neighbor_upscale(map, self.nearest_neighbor_upscale)

                # if self.starting_heightmap is not None:
                #     self.starting_heightmap = tools.arrays.nearest_neighbor_upscale(self.starting_heightmap, self.nearest_neighbor_upscale)

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

        self.clear_downloaded_maps()  # only required to clear the upscale i think
        self.download_image_profile_maps()

        for name in self.image_profiles:
            profile = self.image_profiles[name]
            frame = profile.get_frame()
            tools.images.save(frame, f"{self.folder}/{self.filename_base}.{name}")

        # if self.debug:
        #     self.erosion.debug_update()
        #     tile_count = erosion._width * erosion._height

        #     for attr in dir(self.erosion):
        #         if attr.startswith("_debug"):
        #             value = getattr(self.erosion, attr)

        #             if isinstance(value, float):  # round floats
        #                 value /= float(tile_count)
        #                 print(f"{attr}: {value:.4f}")

        self.save_meta_data()


def main():
    pass


if __name__ == "__main__":
    print("Running main logic...")
    main()
