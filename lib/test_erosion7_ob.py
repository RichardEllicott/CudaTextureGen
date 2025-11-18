"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""

# from tools import *
import tools
import cuda_texture_gen
import imageio.v2 as imageio  # v2 uses numpy arrays
import numpy as np

from typing import Dict, Any


def get_heightmap_01(width=128, height=128):

    octaves = 7
    # octaves = 4 + 1
    # heightmap_scale = 16.0 * 4

    height_map = tools.get_fractal_noise(width=width, height=height, octaves=octaves)
    tools.normalize_array(height_map)

    return height_map


class ErosionRunner:

    erosion = cuda_texture_gen.Erosion7()

    folder = "E:/"
    filename_base = "erosion"

    animation_fps = 5

    build_height_map_animation = True
    build_water_map_animation = True
    build_sediment_map_animation = True
    build_combined_map_animation = True

    frame_count = 64

    steps_per_frame = 8

    combined_map_layout = ["height_map", "water_map", "height_map"]

    nearest_neighbor_upscale = 1  # if > 1 upscale the pixel sizes (makes it easier to see what's going on)

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

    def process(self):

        erosion = self.erosion
        erosion.steps = self.steps_per_frame
        erosion.allocate_device()

        self.save_json()

        def get_mpg_writer(label=""):
            return imageio.get_writer(f"{self.folder}/{self.filename_base}{label}.mp4", fps=self.animation_fps)

        if self.build_height_map_animation:
            height_map_writer = get_mpg_writer()
        if self.build_water_map_animation:
            water_map_writer = get_mpg_writer(".water")
        if self.build_combined_map_animation:
            combined_map_writer = get_mpg_writer(".combined")

        # predeclare
        height_map = erosion.height_map
        water_map = erosion.water_map
        sediment_map = erosion.sediment_map

        for i in range(self.frame_count):

            self.erosion.process()

            height_map = erosion.height_map
            water_map = erosion.water_map
            sediment_map = erosion.sediment_map

            print(f"frame: {i}, height_map min: {height_map.min():.2f}, max: {height_map.max():.2f}")

            tools.normalize_array(height_map)
            # normalize_array(water_map)

            # height_map.clip(0, 10, out=height_map)   # modifies arr directly
            water_map.clip(0, 8, out=water_map)   # modifies arr directly
            sediment_map.clip(0, 1, out=sediment_map)   # modifies arr directly

            # height_map /= heightmap_scale
            # water_map /= heightmap_scale

            if self.nearest_neighbor_upscale > 1:
                height_map = tools.nearest_neighbor_upscale(height_map, self.nearest_neighbor_upscale)
                water_map = tools.nearest_neighbor_upscale(water_map, self.nearest_neighbor_upscale)
                sediment_map = tools.nearest_neighbor_upscale(sediment_map, self.nearest_neighbor_upscale)

            if self.build_height_map_animation:
                height_map_writer.append_data((height_map * 255.0).astype(np.uint8))

            if self.build_water_map_animation:
                water_map_writer.append_data((water_map * 255.0).astype(np.uint8))

            if self.build_combined_map_animation:
                merged_array = tools.merge_numpy_arrays_to_color(height_map, height_map, water_map)
                # merged_array = tools.merge_numpy_arrays_to_color(sediment_map, height_map, water_map)
                combined_map_writer.append_data((merged_array * 255.0).astype(np.uint8))

        metadata = {
            "height_map.min": height_map.min(),
            "height_map.max": height_map.max(),
            "water_map.min": water_map.min(),
            "water_map.max": water_map.max(),
            "sediment_map.min": sediment_map.min(),
            "sediment_map.max": sediment_map.max(),
        }

        tools.save_array_as_image(height_map * 255, self.get_filename_start() + ".height.png")
        tools.save_array_as_image(water_map * 255, self.get_filename_start() + ".water.png")
        tools.save_array_as_image(sediment_map * 255, self.get_filename_start() + ".sediment.png")


# 2D array with shape (0, 0) .... USE FOR CLEARING A HEIGHTMAP.. OPTIONAL
empty_array = np.empty((0, 0), dtype=np.float32)


def test():
    """
    shows okay rivers
    """
    height_map = tools.get_fractal_noise(width=256, height=256, octaves=5)
    tools.normalize_array(height_map)

    runner = ErosionRunner()
    # runner.PRESET_erosion_01()
    runner.PRESET_erosion_02()
    # erosion.PRESET_simple_erosion()

    runner.nearest_neighbor_upscale = 2
    runner.erosion.height_map = height_map * 128.0
    # runner.erosion.height_map = get_heightmap_01() * 32.0


    runner.process()

# test()



def test():
    """
    shows okay rivers
    """
    height_map = tools.get_fractal_noise(width=256, height=256, octaves=5)
    tools.normalize_array(height_map)


    runner = ErosionRunner()
    erosion = runner.erosion


    # runner.PRESET_erosion_01()
    runner.PRESET_erosion_02()
    # erosion.PRESET_simple_erosion()

    runner.nearest_neighbor_upscale = 2

    erosion.height_map = None

    print(type(erosion.height_map))


    test = erosion.height_map

    # runner.erosion.height_map = height_map * 128.0


    # runner.erosion.height_map = get_heightmap_01() * 32.0

    # runner.test321
    # erosion.test_types(None)
    # erosion.test_types(123)
    # erosion.test_types(empty_array)


    # erosion.test321 = None
    # erosion.test321 = 123
    # erosion.test321 = empty_array
    
    # runner.process()

test()
