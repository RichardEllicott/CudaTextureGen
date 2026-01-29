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


from pathlib import Path


class RunnerBase:

    debug: bool = True  # print debug information (could slow down processing)

    erosion = None

    _erosion_default_pars: dict[str, Any]  # filled with the starting pars of erosion

    # output_folder: str = "E:/"  # output folder for movies and images
    # output_folder: str = "./output/"
    output_folder: str = "./godot/cuda_texture_gen/projects/erosion_test/"

    debug_upscale: int = 1 # debug upscale


    filename_base: str = "erosion"  # base string for files

    def get_file_path(self, ext: str) -> Path:
        """
        get file path for saving
        """
        folder = Path(self.output_folder)
        name = f"{self.filename_base}.{ext}"
        return folder / name

    # region MAP_CACHE

    starting_heightmap: NDArray[np.float32] | None = None  # save a starting heightmap

    _image_cache: dict[str, npt.NDArray[np.float32]] = {}  # downloaded maps cache

    def clear_image_cache(self):
        """
        clear cache, call to ensure we donwload a new map
        """
        self._image_cache.clear()

    def download_image(self, name) -> NDArray:
        """
        download a map (if not already in cache)
        """
        if not name in self._image_cache:

            map = getattr(self.erosion, name).array

            if self.debug_upscale > 1:
                map = np.repeat(np.repeat(map, self.debug_upscale, axis=0), self.debug_upscale, axis=1)

            self._image_cache[name] = map
        return self._image_cache[name]


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

    nearest_neighbor_upscale = 1  # if larger than one upscale

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

    def __init__(self, runner: RunnerBase) -> None:
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
                # map = self.runner._maps[channel].copy()
                map = self.runner.download_image(channel).copy()

                # nearest neighbor upscale (maybe move to last?)
                if self.nearest_neighbor_upscale > 1:
                    map = np.repeat(np.repeat(map, self.nearest_neighbor_upscale, axis=0), self.nearest_neighbor_upscale, axis=1)

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


class FP_NormalMap(FrameProfile):

    normal_map: bool = True


    


class Runner(RunnerBase):

    animation_fps: int = 5
    frame_count: int = 64
    process_time: float = 0.0

    # profiles build movies each frame in process
    movie_profiles: dict[str, FrameProfile] = {}
    # profiles build image at end of process
    image_profiles: dict[str, FrameProfile] = {}

    def __init__(self) -> None:

        # self.erosion = cuda_texture_gen.Erosion10()

        self.erosion = cuda_texture_gen.GNC_Erosion()
        # self.erosion = cuda_texture_gen.GNC_Noise()

        self._erosion_default_pars = tools.dicts.from_object(self.erosion)

        self.OUTPUT_PRESET_01()  # defaults

    def OUTPUT_PRESET_01(self):

        self.movie_profiles = {}
        self.image_profiles = {}

        # combined map movie
        movie_profile = FrameProfile(self)
        movie_profile.channels = ["sediment_map", "height_map", "water_map"]  # all channels
        # movie_profile.channels = ["height_map", "height_map", "water_map"]  # height does yellow
        # movie_profile.clip = [None, None, 1.0]
        self.movie_profiles["combined"] = movie_profile

        # height map movie
        movie_profile = FrameProfile(self)
        movie_profile.channels = ["height_map"]
        self.movie_profiles["height"] = movie_profile

        # water map movie
        movie_profile = FrameProfile(self)
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
        movie_profile = FrameProfile(self)
        # movie_profile.channels = ["sediment_map", "height_map", "water_map"]  # all channels
        movie_profile.channels = ["height_map", "height_map", "water_map"]  # height does yellow
        # movie_profile.clip = [None, None, 1.0]
        self.movie_profiles["combined"] = movie_profile

    def OUTPUT_PRESET_03(self):

        self.OUTPUT_PRESET_01()
        # combined map movie
        movie_profile = FrameProfile(self)
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

    def save_meta_data(self):

        file_path = self.get_file_path(ext="data.json")

        data = {}

        data["process_time"] = self.process_time

        data["erosion_settings"] = self.get_erosion_pars()

        runner_settings = {}
        runner_settings["frame_count"] = self.frame_count
        data["runner_settings"] = runner_settings

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _compute(self):
        """
        hook to override
        """
        self.erosion.compute()

    def process(self) -> None:
        """

        """

        start_time = time.perf_counter()

        # save starting heightmap
        if not self.starting_heightmap:
            # self.starting_heightmap = self.erosion.height_map.array
            # if self.starting_heightmap is not None and self.starting_heightmap.size > 0:
            #     tools.images.save(
            #         tools.arrays.normalized(self.starting_heightmap),
            #         self.get_file_path(ext="start.png")
            #     )
            pass

        # create movie writers
        movie_writers = {}
        for key in self.movie_profiles:
            file_path = self.get_file_path(ext=f"{key}.mp4")
            writer = imageio.get_writer(file_path, fps=self.animation_fps)
            movie_writers[key] = writer

        for i in range(self.frame_count):

            self.clear_image_cache()  # ensures new maps are downloaded

            self._compute()

            # for each movie writer write using frame profile
            for key in movie_writers:
                profile = self.movie_profiles[key]
                frame = profile.get_frame()
                writer = movie_writers[key]
                writer.append_data((frame * 255.0).astype(np.uint8))

        end_time = time.perf_counter()
        self.process_time = end_time - start_time

        print(f"process time: {self.process_time:.3f} seconds")

        for name in self.image_profiles:
            profile = self.image_profiles[name]
            frame = profile.get_frame()

            tools.images.save(frame, self.get_file_path(ext=name))

        self.save_meta_data()


class ErosionRunnerGNC(Runner):

    pass


class ErosionRunnerDelta(Runner):

    def __init__(self) -> None:

        self.erosion = cuda_texture_gen.GNC_ErosionDelta()
        self._erosion_default_pars = tools.dicts.from_object(self.erosion)
        self.OUTPUT_PRESET_01()  # defaults


def main():
    pass


if __name__ == "__main__":
    print("Running main logic...")
    main()
