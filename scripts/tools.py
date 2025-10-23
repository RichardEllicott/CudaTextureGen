"""

"""
import numpy as np
from PIL import Image

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
    
    
