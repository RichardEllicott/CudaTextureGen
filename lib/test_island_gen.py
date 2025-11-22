"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""
import tools
import cuda_texture_gen



def get_island_template(width: int, height: int):
    print("get_island_template...")


    island = tools.circle_array(128, 128, 40)
    # island = tools.load_image_as_array("./images/island03.png")


    island = tools.resize_array_2d(island, width, height)
    # island = tools.blur_array(island, 8)

    warp_strength = 0.125
    noise_generator = cuda_texture_gen.NoiseGenerator()
    noise_generator.period = 3
    noise1 = noise_generator.generate(width, height)
    noise_generator.seed += 1
    noise2 = noise_generator.generate(width, height)


    resample = cuda_texture_gen.Resample()
    resample.input = island
    resample.map_x = noise1 * warp_strength
    resample.map_y = noise2 * warp_strength
    resample.process()
    island = resample.output


    island = tools.resize_array_2d(island, 512, 512)

    tools.save_array_as_image(island * 255, "./output/test_island_template.png")


get_island_template(128, 128)