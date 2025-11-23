"""

new device array test

"""
import tools
import cuda_texture_gen


def test_device_array_n():
    print("test_device_array_n...")

    cuda_texture_gen.test_device_array_n()

    # noise = tools.fractal_noise(128, 128)
    noise = tools.fractal_noise_rgb(128, 128)

    tools.save_array_as_image(noise * 255, "./output/noise.png")

    template = cuda_texture_gen.TemplateDArray1()


    template.device_array_n3d_test = noise
    noise2 = template.device_array_n3d_test
    tools.save_array_as_image(noise2 * 255, "./output/noise2.png")





test_device_array_n()
