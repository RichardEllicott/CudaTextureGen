"""
image helpers
"""
import numpy as np
import imageio.v2 as imageio


def save_array_as_image(array: np.ndarray, filename) -> None:
    """
    Save a NumPy 2D array as an image.
    Supports .png (uint8) or .tif/.tiff (float32).
    """

    # Handle multiple filenames
    if isinstance(filename, (list, tuple)):
        for fname in filename:
            save_array_as_image(array, fname)
        return

    ext = filename.lower()

    if ext.endswith(".png"):
        # PNG expects 8-bit integers
        imageio.imwrite(filename, array.astype(np.uint8))

    elif ext.endswith((".tif", ".tiff")):
        # TIFF supports float32 (good for heightmaps)
        imageio.imwrite(filename, array.astype(np.float32))

    else:
        raise ValueError(f"Unsupported format: {filename}")


def load_array_from_image(filename) -> np.ndarray:
    """
    Load an image file into a NumPy array.
    Preserves whatever dimensionality the image has:
    - Grayscale → 2D array (H, W)
    - RGB → 3D array (H, W, 3)
    - RGBA → 3D array (H, W, 4)
    - Float TIFFs → 2D or 3D depending on channels
    """
    array = imageio.imread(filename)
    return array.astype(np.float32)
