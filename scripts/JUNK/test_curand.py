"""




"""

from tools import *
import cuda_texture_gen



def test_curand():

    print("test_curand...")
    print(dir(cuda_texture_gen))

    template_class3 = cuda_texture_gen.TemplateClass3()

    print(dir(template_class3))

    array = get_fractal_noise()

    template_class3.image = array
    template_class3.process()
    array = template_class3.image

    # array = template_class3.process(array)
    print("height range: [{}, {}]".format(array.min(), array.max()))

    save_array_as_image(array * 255, "output/xxx_test.png")

test_curand()

