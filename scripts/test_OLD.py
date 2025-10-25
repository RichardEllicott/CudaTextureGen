"""

🚧 testing the framework 🚧


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
import python_bootstrap # bootstrap to our fresh compiled module
import inspect  # function reflect
from PIL import Image
import numpy as np
from pathlib import Path
import cuda_texture_gen
#


# print that puts a snake at front so we can see the print from python!
def snake_print(*args, **kwargs):
    builtins.print("🐍", *args, **kwargs)

print = snake_print

def print_current_function():
    frame = inspect.currentframe().f_back  # caller's frame
    func_name = frame.f_code.co_name
    args, _, _, values = inspect.getargvalues(frame)
    arg_str = ', '.join(f"{arg}={values[arg]!r}" for arg in args)
    print(f"{func_name}({arg_str})...")



def test_cuda_hello():
    print_current_function()
    cuda_texture_gen.cuda_hello()


# save numpy 2d array as an image (supports .png or .tif)
def save_array_as_image(arr, filename):
    print_current_function()

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
    


def load_array_from_image(filename):
    print_current_function()
    img = Image.open(filename).convert("L")
    arr = np.array(img, dtype=np.float32)
    return arr

# normalize array in place (make from 0 to 1)
def normalize_array(arr):
    print_current_function()

    arr -= arr.min()
    arr /= arr.max()

# offset array by half (to test tiling)
def offset_array(arr):
    print_current_function()

    # Compute half offsets
    dx = arr.shape[1] // 2  # width
    dy = arr.shape[0] // 2  # height

    # Apply toroidal (wraparound) shift
    # shifted = np.roll(arr, shift=(dy, dx), axis=(0, 1))
    arr[:] = np.roll(arr, shift=(dy, dx), axis=(0, 1))






def test_noise_generator(width=256, height=256, filename="output/noise_gen_test256.png", type=0):
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    arr = np.zeros((width, height), dtype=np.float32)
    gen = cuda_texture_gen.NoiseGenerator()

    gen.type = type

    gen.period = 9
    gen.period = 17

    gen.seed = 0

    gen.fill(arr)  # generate the noise

    offset_arr(arr)  # offset to check the tiling

    print("height range: [{}, {}]".format(arr.min(), arr.max()))

    # arr = (arr - arr.min()) / (arr.max() - arr.min())  # normalize to [0, 1]
    arr = arr * 0.5 + 0.5  # [-1, 1] => [0, 1]

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

    cuda_texture_gen.blur(arr, amount=15, wrap=True)

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
    erosion = cuda_texture_gen.ErosionSimulator()

    # good settings
    erosion.erosion_rate = 0.01
    erosion.deposition_rate = 0.02 * 0.5
    # lowers squareness if higher as makes sure works on steeper slopes
    erosion.slope_threshold = 0.01
    erosion.steps = 512 * 4
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

    offset_array(arr)

    print("height range: [{}, {}]".format(arr.min(), arr.max()))

    save_array_as_image(arr, output_filename)


## generate every noise type
def test_all_noise():
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    os.makedirs("output", exist_ok=True)

    for t in cuda_texture_gen.NoiseGenerator.Type:
        print(t.name, t.value)
        test_noise_generator(
            1024, 1024, "output/noise_{}.png".format(t.name), t.value)



def test_warped_noise():
    raise ValueError("test_warped_noise not iimplemented!")
    print('{}()...'.format(inspect.currentframe().f_code.co_name))
    gen = cuda_texture_gen.NoiseGenerator()




# gen some gradient noise, and make a normal map and ao map
def test_shader_maps(filename):
    print('{}()...'.format(inspect.currentframe().f_code.co_name))

    for t in cuda_texture_gen.NoiseGenerator.Type:
        print(t.name, t.value)

    test_noise_generator(512, 512, filename,
                         cuda_texture_gen.NoiseGenerator.Type.Gradient2D.value)

    shader_maps = cuda_texture_gen.ShaderMaps()

    print(shader_maps)
    print(dir(shader_maps))

    print("load image...")
    img = Image.open(filename).convert("L")
    arr = np.array(img, dtype=np.float32)

    normal_arr = shader_maps.generate_normal_map(arr)
    ao_arr = shader_maps.generate_ao_map(arr * 0.5, radius=2)

    path = Path(filename)
    save_array_as_image(
        normal_arr * 255.0, str(path.with_name(path.stem + ".normal" + path.suffix)))
    save_array_as_image(
        ao_arr * 255.0, str(path.with_name(path.stem + ".ao" + path.suffix)))



def new_noise_test(width = 256, height = 256, filename = "output/new_noise_test.png"):

    noise_generator = cuda_texture_gen.NoiseGenerator()

    noise_generator.type = 1
    noise_generator.period = 9
    # noise_generator.period = 17
    noise_generator.seed = 0

    # arr = np.zeros((width, height), dtype=np.float32) 
    # noise.fill(arr)
    arr = noise_generator.generate(256, 256) # create new noise (returns new array)

    arr = (arr - arr.min()) / (arr.max() - arr.min())  # normalize to [0, 1]
    # arr = arr * 0.5 + 0.5  # [-1, 1] => [0, 1]


    print("height range: [{}, {}]".format(arr.min(), arr.max()))
    # save_array_as_image(arr * 255, filename)


    shader_maps = cuda_texture_gen.ShaderMaps()

    arr2 = shader_maps.generate_ao_map(arr * 40, 2.0)


    save_array_as_image(arr2 * 255, filename)




# test_all_noise()


# # test_cuda_hello()
# os.makedirs("output", exist_ok=True)
# # # GENERATE NOISE AND ERODE
noise_filename = "output/noise.png"
test_noise_generator(1024, 1024, noise_filename, 1)
test_errosion(noise_filename, "output/erode.png")
# test_blur(noise_filename, "output/blur.png")


# print(dir(cuda_hello))

# test_shader_maps(noise_filename)



# new_noise_test()



def template_class_test():
    template_class = cuda_texture_gen.TemplateClass()
    print(template_class)
    print(dir(template_class))
# template_class_test()