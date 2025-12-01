"""

Substrata

-project name??

"""


import tools
# import cuda_texture_gen


subfolder = "./output/"


def get_island():

    from IslandGenerator import IslandGenerator
    island_generator = IslandGenerator()

    island_generator.height *= 2 * 2
    island_generator.width *= 2 * 2
    island_generator.diameter *= 2

    island_generator.pre_blur = 24.0
    # island_generator.blur = 24.0

    island_generator.warp_octaves = 3

    island = island_generator.island

    tools.images.save(island, f"{subfolder}layer_maker.island.png")

    return island


def test_gradient_object():
    print("test_gradient_object...")

    # gradient = tools.gradients.get_test_gradient_01()
    gradient = tools.gradients.get_test_gradient_02(0, 8)

    tools.images.save(gradient.get_data(), f"{subfolder}layer_maker.gradient.png")

    gradient_strip = gradient.render(128)
    # tools.print_array_information(gradient_strip)
    tools.palettes.save(gradient_strip, f"{subfolder}layer_maker.gradient.palette.png")

    island = get_island()

    island_rgb = tools.palettes.apply_gradient_strip(island, gradient_strip)
    tools.images.save(island_rgb, f"{subfolder}layer_maker.island.rgb.png")


def main():
    test_gradient_object()


if __name__ == "__main__":
    print("Running main logic...")
    main()  # This block runs only if executed directly
