"""
"""
import numpy as np
from numpy.typing import NDArray
from .image_io_helpers import *


def save_palette(gradient: NDArray[np.float32], filename: str) -> None:
    """
    Save a palette (N,3) as a vertical strip image (N,1,3).
    """
    gradient = gradient.reshape((-1, 1, 3))
    save_image(gradient, filename)


def load_palette(filename: str) -> NDArray[np.float32]:
    """
    Load a palette strip image (N,1,3) and collapse back to (N,3).
    """
    gradient = load_image(filename)
    return gradient.reshape((-1, 3))


def apply_palette(heightmap: NDArray[np.float32],
                  palette: NDArray[np.float32],
                  smooth: bool = False) -> NDArray[np.float32]:
    """
    Map a grayscale heightmap to a color image using a palette.

    Parameters
    ----------
    heightmap : NDArray[np.float32]
        2D array of grayscale values (any range).
    palette : NDArray[np.float32]
        Array of shape (N,3) with RGB colors in [0,1].
    smooth : bool, default False
        If False, discrete mapping (snap to nearest palette color).
        If True, smooth linear interpolation between adjacent colors.

    Returns
    -------
    NDArray[np.float32]
        Colorized image of shape (H,W,3) with values in [0,1].
    """
    # Normalize heightmap to [0,1]
    hmap = heightmap.astype(np.float32)
    hmap -= hmap.min()
    if hmap.max() > 0:
        hmap /= hmap.max()

    n_colors = len(palette)

    if not smooth:
        # Discrete mapping
        indices = (hmap * (n_colors - 1)).astype(np.int32)
        colored = palette[indices]
    else:
        # Smooth interpolation
        scaled = hmap * (n_colors - 1)
        idx_low = np.floor(scaled).astype(np.int32)
        idx_high = np.clip(idx_low + 1, 0, n_colors - 1)
        t = scaled - idx_low
        colored = (1 - t)[..., None] * palette[idx_low] + t[..., None] * palette[idx_high]

    return colored
