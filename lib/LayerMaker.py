"""

Substrata

-project name??

"""
from typeguard import typechecked



from typing import Union, Sequence
import tools
import cuda_texture_gen
from numpy.typing import NDArray
from typing import Any, Optional
import numpy as np
import colour  # pip install colour-science
from matplotlib.colors import to_rgb
import numpy as np

@typechecked
class LayerMaker():

    input = None
    gradient = None

    def process():

        pass


# substrata

def test2():

    subfolder = "./output/"

    from IslandGenerator import IslandGenerator
    island_generator = IslandGenerator()
    island = island_generator.island
    island = tools.blur_array(island, 12.0)
    tools.save_image(island, f"{subfolder}layer_maker.height.png")

    random_gradient = tools.gradients.random_gradient(64, 512)
    # for entry in random_gradient:
    #     print(entry)

    random_gradient = tools.gradients.tuples_gradient_to_gradient_strip(random_gradient)

    tools.save_image(random_gradient, f"{subfolder}layer_maker.gradient.png")


subfolder = "./output/"


def get_island():

    from IslandGenerator import IslandGenerator
    island_generator = IslandGenerator()

    island_generator.height *= 2 * 2
    island_generator.width *= 2 * 2
    island_generator.diameter *= 2

    island_generator.pre_blur = 24.0
    # island_generator.blur = 24.0

    island_generator.noise_octaves = 3


    island = island_generator.island
    
    tools.save_image(island, f"{subfolder}layer_maker.island.png")

    return island


def test_gradient_object():
    print("test_gradient_object...")

    # gradient = tools.gradients.get_test_gradient_01()
    gradient = tools.gradients.get_test_gradient_02(0, 8)

    tools.save_image(gradient.data(), f"{subfolder}layer_maker.gradient.png")

    gradient_strip = gradient.render(128)
    # tools.print_array_information(gradient_strip)
    tools.palettes.save(gradient_strip, f"{subfolder}layer_maker.gradient.palette.png")

    island = get_island()


    island_rgb = tools.palettes.apply_gradient_strip(island, gradient_strip)
    tools.save_image(island_rgb, f"{subfolder}layer_maker.island.rgb.png")


def main():
    test_gradient_object()


if __name__ == "__main__":
    print("Running main logic...")
    main()  # This block runs only if executed directly
