"""

"""
import numpy as np
from PIL import Image
import inspect


# class Tools:

# print function call
def print_current_function():
    frame = inspect.currentframe().f_back  # caller's frame
    func_name = frame.f_code.co_name
    args, _, _, values = inspect.getargvalues(frame)
    arg_str = ', '.join(f"{arg}={values[arg]!r}" for arg in args)
    print(f"{func_name}({arg_str})...")

# save numpy 2d array as an image (supports .png or .tif)
def save_array_as_image(arr, filename):

    if isinstance(filename, (list, tuple)):
        for fname in filename:
            self.save_array_as_image(arr, fname)
        return

    if filename.endswith(".png"):
        img = Image.fromarray(arr.astype(np.uint8))
        img.save(filename)

    elif filename.endswith((".tif", ".tiff")):
        img = Image.fromarray(arr.astype(np.float32))
        img.save(filename)

    else:
        raise ValueError(f"Unsupported format: {filename}")

# load image as black and white array
def load_array_from_image(filename):
    img = Image.open(filename).convert("L")
    arr = np.array(img, dtype=np.float32)
    return arr

# normalize array in place (make from 0 to 1)
def normalize_array(arr):
    arr -= arr.min()
    arr /= arr.max()

# offset array by half (to test tiling)
def offset_array(arr):

    # Compute half offsets
    dx = arr.shape[1] // 2  # width
    dy = arr.shape[0] // 2  # height

    # Apply toroidal (wraparound) shift
    # shifted = np.roll(arr, shift=(dy, dx), axis=(0, 1))
    arr[:] = np.roll(arr, shift=(dy, dx), axis=(0, 1))
