"""

array helpers

"""

from scipy.ndimage import zoom, gaussian_filter
import numpy as np
from numpy.typing import NDArray, DTypeLike
from typing import Any


def print_array_information(array: NDArray) -> None:
    """
    print debug info about array
    """
    print("🐈 ", type(array))
    print("shape:", array.shape)        # dimensions
    print("dtype:", array.dtype)        # element type
    print("ndim:", array.ndim)          # number of dimensions
    print("strides:", array.strides)    # byte steps between elements
    print("flags:", array.flags)        # contiguity info


def normalize(array: NDArray[np.floating]) -> None:
    """
    Normalize array in place to [0,1].
    If all values are equal, returns zeros.
    """
    min_val = array.min()
    max_val = array.max()
    range_val = max_val - min_val

    if range_val == 0:  # guard against div by zero
        array[:] = 0.0
    else:
        array -= min_val
        array /= range_val


def normalized(array: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    return a normalized array (don't change the input)
    """
    result = array.copy()
    normalize(result)
    return result


def map_range(array: NDArray[np.floating],
              new_min: float = 0.0,
              new_max: float = 1.0) -> None:
    """
    Map array values in place from [min(array), max(array)] to [new_min, new_max].
    If all values are equal, fills with new_min.
    """
    old_min = array.min()
    old_max = array.max()
    old_range = old_max - old_min

    if old_range == 0:  # guard against division by zero
        array[:] = new_min
    else:
        scale = (new_max - new_min) / old_range
        array[:] = (array - old_min) * scale + new_min


def offset(array: np.ndarray, x_offset: float = 0.5, y_offset: float = 0.5) -> None:
    """
    Offset array toroidally (wraparound) by fractional amounts of width/height.

    Parameters
    ----------
    array : np.ndarray
        2D array to shift in place.
    x_offset : float
        Fraction of width to shift (0.5 = half width). Default 0.5.
    y_offset : float
        Fraction of height to shift (0.5 = half height). Default 0.5.
    """
    h, w = array.shape[:2]

    # Convert fractional offsets to integer pixel shifts
    dx = int(round(w * x_offset)) % w
    dy = int(round(h * y_offset)) % h

    # Apply toroidal shift
    array[:] = np.roll(array, shift=(dy, dx), axis=(0, 1))


def merge_to_color(
    red: NDArray[Any] | None = None,
    green: NDArray[Any] | None = None,
    blue: NDArray[Any] | None = None,
    alpha: NDArray[Any] | None = None,
    shape: tuple[int, ...] | None = None,
    dtype: DTypeLike = np.float32,  # default float until you clamp/cast
) -> NDArray[Any]:
    """
    Merge separate Red, Green, Blue (and optional Alpha) numpy arrays into a single image.
    """
    if shape is None:
        for ch in (red, green, blue, alpha):
            if ch is not None:
                shape = ch.shape
                dtype = ch.dtype
                break
    if shape is None:
        raise ValueError("Must provide at least one channel or a shape")

    def ensure_channel(ch: NDArray[Any] | None, fill_value: int) -> NDArray[Any]:
        if ch is None:
            return np.full(shape, fill_value, dtype=dtype)
        if ch.shape != shape:
            raise ValueError("Channel shape mismatch")
        return ch

    red = ensure_channel(red, 0)
    green = ensure_channel(green, 0)
    blue = ensure_channel(blue, 0)

    if alpha is None:
        return np.stack([red, green, blue], axis=-1)
    else:
        alpha = ensure_channel(alpha, 0)
        return np.stack([red, green, blue, alpha], axis=-1)


def nearest_neighbor_upscale(array: NDArray, factor: int) -> NDArray:
    """
    ⚠️ so minimal, kept as notes!
    scale an image up in size with no filter, this is like what you might do for pixel art
    it is useful to visualize the erosion better for debugging
    """
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def tile(array: NDArray, repeat_x: int, repeat_y: int) -> NDArray:
    """
    ⚠️ so minimal, kept as notes!
    tile the array, so repeat the same to maybe test seamless etc
    """
    return np.tile(array, (repeat_x, repeat_y))


# scipy
def resize(array: NDArray[np.floating], width: int, height: int, order: int = 1):
    """
    Resize a 2D NumPy array with resampling using scipy.ndimage.zoom.
    """
    zoom_factors = (height / array.shape[0], width / array.shape[1])
    return zoom(array, zoom_factors, order=order)


# scipy
def blur(array: NDArray[np.floating], sigma: float) -> NDArray[np.floating]:
    """
    blur array with scipy
    """

    return gaussian_filter(array, sigma=sigma)


def circle(width: int, height: int, radius: int) -> NDArray[np.float32]:
    """
    Create a 2D float32 NumPy array with a filled circle in the middle.
    Values: 1.0 inside the circle, 0.0 outside.
    """
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    dist_sq = (x - cx)**2 + (y - cy)**2
    mask = dist_sq <= radius**2

    arr = np.zeros((height, width), dtype=np.float32)
    arr[mask] = 1.0
    return arr
