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


test_cuda_hello()