# add the build/python directory


import sys
import os
import platform # detect platform
import inspect # function reflect

# test add the folder for python
root_dir = os.path.dirname(__file__)
if platform.system() == "Windows":
    build_subdir = os.path.join("build", "windows")
else:
    build_subdir = os.path.join("build", "linux")
sys.path.append(os.path.join(root_dir, build_subdir, "python"))

import cuda_hello


# snake print
import builtins
def snake_print(*args, **kwargs):
    builtins.print("🐍", *args, **kwargs)
print = snake_print



# Ubuntu Linux no longer wants to use PIP, but apt
# sudo apt install python3-pil
# sudo apt install python3-numpy

from PIL import Image
import numpy as np

# img = Image.open("perlin_noise_example.png").convert("L")  # grayscale for simplicity
img = Image.new(mode="RGB", size=(128, 128))
arr = np.array(img, dtype=np.float32)         # shape (H, W)



def test_cuda_hello():
    print('{}()...'.format(inspect.currentframe().f_code.co_name))
    cuda_hello.cuda_hello()




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




# offset array by half (to test tiling)
def offset_arr(arr):
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    # Compute half offsets
    dx = arr.shape[1] // 2  # width
    dy = arr.shape[0] // 2  # height

    # Apply toroidal (wraparound) shift
    # shifted = np.roll(arr, shift=(dy, dx), axis=(0, 1))
    arr[:] = np.roll(arr, shift=(dy, dx), axis=(0, 1))





def test_c_noise_generation(width=256, height=256, filename="output/noise_gen_test256.png"):
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    arr = np.zeros((width, height), dtype=np.float32)
    gen = cuda_hello.CNoiseGenerator()


    gen.period = 9
    # gen.period = 17


    gen.seed = 42

    gen.fill(arr)

    print("height range: [{}, {}]".format(arr.min(), arr.max()))


    offset_arr(arr)

    arr = (arr - arr.min()) / (arr.max() - arr.min())  # normalize to [0, 1]

    # arr *= 255.0
    save_array_as_image(arr * 255, filename)
    save_array_as_image(arr, filename + '.tif')



def test_errosion(filename, output_filename):

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

# test_cuda_hello()


# # GENERATE NOISE AND ERODE
noise_filename = "output/noise.png"
test_c_noise_generation(512, 512, noise_filename)


test_errosion(noise_filename, "output/erode_test.png")

