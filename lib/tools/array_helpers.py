"""

array helpers

"""

import numpy as np


def normalize_array(array: np.ndarray) -> None:
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


def offset_array(array: np.ndarray, x_offset: float = 0.5, y_offset: float = 0.5) -> None:
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


def merge_numpy_arrays_to_color(
    red: np.ndarray = None,
    green: np.ndarray = None,
    blue: np.ndarray = None,
    alpha: np.ndarray = None,
    shape: tuple = None,
    dtype: np.dtype = np.uint8
) -> np.ndarray:
    """
    Merge separate Red, Green, Blue (and optional Alpha) numpy arrays into a single image.

    Missing channels default to black (0). If Alpha is not provided, the result
    will be an RGB image with 3 channels. If Alpha is provided, the result will
    be an RGBA image with 4 channels.

    Parameters
    ----------
    red, green, blue, alpha : np.ndarray or None
        Arrays of the same shape, representing channels. Any can be None.
        - red   : Red channel
        - green : Green channel
        - blue  : Blue channel
        - alpha : Alpha channel (optional)
    shape : tuple or None
        Shape to use if some channels are None. If None, inferred from the first non-None channel.
    dtype : np.dtype
        Data type of the output array (default uint8).

    Returns
    -------
    image : np.ndarray
        Combined array of shape (H, W, 3) if alpha is None,
        or (H, W, 4) if alpha is provided.
    """
    # infer shape from first non-None channel
    if shape is None:
        for ch in (red, green, blue, alpha):
            if ch is not None:
                shape = ch.shape
                dtype = ch.dtype
                break
    if shape is None:
        raise ValueError("Must provide at least one channel or a shape")

    def ensure_channel(ch, fill_value):
        if ch is None:
            return np.full(shape, fill_value, dtype=dtype)
        if ch.shape != shape:
            raise ValueError("Channel shape mismatch")
        return ch

    red = ensure_channel(red, 0)
    green = ensure_channel(green, 0)
    blue = ensure_channel(blue, 0)

    if alpha is None:
        # Return RGB only
        return np.stack([red, green, blue], axis=-1)
    else:
        alpha = ensure_channel(alpha, 0)
        return np.stack([red, green, blue, alpha], axis=-1)


def nearest_neighbor_upscale(array, factor):
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def tile_array_2d(array, repeat_x, repeat_y):
    return np.tile(array, (repeat_x, repeat_y))