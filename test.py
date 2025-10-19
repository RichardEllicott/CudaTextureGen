"""
Ubuntu Linux no longer wants to use PIP, but apt:
    sudo apt install python3-pil
    sudo apt install python3-numpy

Windows still uses PIP:
    pip install Pillow
    pip install numpy


update on windows:

pip list --outdated --format=json | ConvertFrom-Json | ForEach-Object {
    pip install --upgrade $_.name
}

    

"""


import builtins  # snake print
import os
import platform
import sys
import cuda_hello_bootstrap
import inspect  # function reflect
from PIL import Image
import numpy as np
import cuda_hello

# fun print that puts a snake at front!


def snake_print(*args, **kwargs):
    builtins.print("🐍", *args, **kwargs)


print = snake_print


def test_cuda_hello():
    print('{}()...'.format(inspect.currentframe().f_code.co_name))
    cuda_hello.cuda_hello()


# save numpy 2d array as an image (supports .png or .tif)
def save_array_as_image(arr, filename):

    if isinstance(filename, (list, tuple)):
        for fname in filename:
            save_array_as_image(arr, fname)
        return

    if filename.endswith(".png"):
        img = Image.fromarray(arr.astype(np.uint8))
        img.save(filename)

    elif filename.endswith((".tif", ".tiff")):
        img = Image.fromarray(arr.astype(np.float32))
        img.save(filename)

    else:
        raise ValueError(f"Unsupported format: {filename}")


# simple tests
def test_python_object_creating():

    arr = cuda_hello.get_2d_numpy_array(1024, 1024)
    print(arr)

    arr = cuda_hello.get_list_of_lists(12, 12)
    print(arr)


# offset array by half (to test tiling)
def offset_arr(arr):
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    # Compute half offsets
    dx = arr.shape[1] // 2  # width
    dy = arr.shape[0] // 2  # height

    # Apply toroidal (wraparound) shift
    # shifted = np.roll(arr, shift=(dy, dx), axis=(0, 1))
    arr[:] = np.roll(arr, shift=(dy, dx), axis=(0, 1))

# seamless noise


def test_noise_generator_d(width=256, height=256, filename="output/noise_gen_test256.png", type=0):
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    arr = np.zeros((width, height), dtype=np.float32)
    gen = cuda_hello.NoiseGeneratorD()

    gen.type = type

    gen.period = 9
    gen.period = 17

    gen.seed = 0

    gen.fill(arr) # generate the noise

    offset_arr(arr) # offset to check the tiling

    print("height range: [{}, {}]".format(arr.min(), arr.max()))

    # arr = (arr - arr.min()) / (arr.max() - arr.min())  # normalize to [0, 1]
    arr = arr * 0.5 + 0.5 # [-1, 1] => [0, 1]

    print("height range: [{}, {}]".format(arr.min(), arr.max()))


    save_array_as_image(arr * 255, filename)
    # save_array_as_image(arr, filename + '.tif')


def test_blur(filename, output_filename):
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    print("load image...")
    img = Image.open(filename).convert("L")
    arr = np.array(img, dtype=np.float32)
    print("height range: [{}, {}]".format(arr.min(), arr.max()))
    print("blur...")

    cuda_hello.blur(arr, amount=15, wrap=True)

    save_array_as_image(arr, output_filename)


# errosion
def test_errosion(filename, output_filename):
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    print("load image...")
    img = Image.open(filename).convert("L")
    arr = np.array(img, dtype=np.float32)
    print("height range: [{}, {}]".format(arr.min(), arr.max()))

    print("scale image...")
    arr /= 255.0
    print("height range: [{}, {}]".format(arr.min(), arr.max()))

    print("erode...")
    erosion = cuda_hello.ErosionSimulator()

    # good settings
    erosion.erosion_rate = 0.01
    erosion.deposition_rate = 0.02 * 0.5
    # lowers squareness if higher as makes sure works on steeper slopes
    erosion.slope_threshold = 0.01
    erosion.steps = 512
    erosion.jitter = 0.0

    erosion.steps = 256
    erosion.steps = 128

    # erosion.sediment_transport_rate = 1.0 # implicit

    # erosion.jitter = 0.0

    # erosion.min_height = 0.0

    erosion.run_erosion(arr)

    arr = np.clip(arr, None, 1.0)

    print("height range: [{}, {}]".format(arr.min(), arr.max()))

    print("normalize and scale image...")
    arr -= arr.min()
    arr /= arr.max()
    arr *= 255.0

    offset_arr(arr)

    print("height range: [{}, {}]".format(arr.min(), arr.max()))

    save_array_as_image(arr, output_filename)


def test_all_noise():
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    os.makedirs("output", exist_ok=True)
    # noise_filename = "output/noise.png"


    for t in cuda_hello.NoiseGeneratorD.Type:
        print(t.name, t.value)
        test_noise_generator_d(1024, 1024, "output/noise_{}.png".format(t.name), t.value)



    # for tyoe in cuda_hello.NoiseGeneratorD.Type:
    #     # print(dir(type))
    #     print(type.__name__)


    # for type in range(5+2):

    #     print("noise type: {}".format(type))

    #     
    #     print()



def test_warped_noise():
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    gen = cuda_hello.NoiseGeneratorD()

    




# test_all_noise()
test_warped_noise()



# # test_cuda_hello()
# os.makedirs("output", exist_ok=True)

# # # GENERATE NOISE AND ERODE
# noise_filename = "output/noise.png"
# test_c_noise_generation(512, 512, noise_filename)
# # test_errosion(noise_filename, "output/erode.png")
# test_blur(noise_filename, "output/blur.png")


# print(dir(cuda_hello))
