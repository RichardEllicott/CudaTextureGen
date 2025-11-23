"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""
import tools
import cuda_texture_gen


import math


def prove_noise_offset():
    print("prove_noise_offset...")

    noise_gen = cuda_texture_gen.Noise()

    noise = noise_gen.noise
    print(f"noise min: {noise.min():.3f}, max: {noise.max():.3f}")
    tools.normalize_array(noise)
    # tools.offset_array(noise)
    tools.save_array_as_image(noise * 255, "./output/noise.png")

    noise_gen.x = 0.5
    noise_gen.y = 0.5

    noise2 = noise_gen.noise
    print(f"noise2 min: {noise2.min():.3f}, max: {noise2.max():.3f}")
    tools.normalize_array(noise2)
    tools.offset_array(noise2)
    tools.save_array_as_image(noise2 * 255, "./output/noise2.png")


# prove_noise_offset()


# seamless broke with rotation
# https://claude.ai/chat/5cb3ec32-d1a7-444e-bcea-446f7505b7a9
def prove_noise_rotation():

    print("prove_noise_rotation...")

    noise_gen = cuda_texture_gen.Noise()

    noise = noise_gen.noise
    print(f"noise min: {noise.min():.3f}, max: {noise.max():.3f}")
    tools.normalize_array(noise)
    # tools.offset_array(noise)
    tools.save_array_as_image(noise * 255, "./output/noise.png")

    # noise_gen.x = 0.5
    # noise_gen.y = 0.5

    # noise_gen.rotate_x = math.radians(45.0)
    # noise_gen.rotate_y = math.radians(5.0)
    noise_gen.rotate_z = math.radians(10.0)

    noise2 = noise_gen.noise
    print(f"noise2 min: {noise2.min():.3f}, max: {noise2.max():.3f}")
    tools.normalize_array(noise2)
    # tools.offset_array(noise2)
    tools.save_array_as_image(noise2 * 255, "./output/noise2.png")


    final = tools.merge_numpy_arrays_to_color(noise, noise2, None)
    tools.save_array_as_image(final * 255, "./output/noise.rgb.png")



prove_noise_rotation()

# def get_island_template(width: int, height: int):
#     print("get_island_template...")


#     island = tools.circle_array(128, 128, 40)
#     # island = tools.load_image_as_array("./images/island03.png")


#     island = tools.resize_array_2d(island, width, height)
#     # island = tools.blur_array(island, 8)

#     warp_strength = 0.125
#     noise_generator = cuda_texture_gen.NoiseGenerator()
#     noise_generator.period = 3
#     noise1 = noise_generator.generate(width, height)
#     noise_generator.seed += 1
#     noise2 = noise_generator.generate(width, height)


#     resample = cuda_texture_gen.Resample()
#     resample.input = island
#     resample.map_x = noise1 * warp_strength
#     resample.map_y = noise2 * warp_strength
#     resample.process()
#     island = resample.output


#     island = tools.resize_array_2d(island, 512, 512)

#     tools.save_array_as_image(island * 255, "./output/test_island_template.png")


# get_island_template(128, 128)
