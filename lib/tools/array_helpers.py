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


def merge_numpy_arrays_to_rgba(r=None, g=None, b=None, a=None, shape=None, dtype=np.uint8) -> np.ndarray:
    """
    Merge separate R, G, B (and optional A) numpy arrays into a single RGBA image.
    Missing channels default to black (0). Alpha defaults to fully opaque.

    Parameters
    ----------
    r, g, b, a : np.ndarray or None
        Arrays of the same shape, representing channels. Any can be None.
    shape : tuple or None
        Shape to use if some channels are None. If None, inferred from the first non-None channel.
    dtype : np.dtype
        Data type of the output array (default uint8).

    Returns
    -------
    rgba : np.ndarray
        Combined array of shape (H, W, 4).
    """
    # infer shape from first non-None channel
    if shape is None:
        for ch in (r, g, b, a):
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

    r = ensure_channel(r, 0)
    g = ensure_channel(g, 0)
    b = ensure_channel(b, 0)

    if a is None:
        # opaque: max for integers, 1.0 for floats
        if np.issubdtype(dtype, np.integer):
            fill = np.iinfo(dtype).max
        else:
            fill = 1.0
        a = np.full(shape, fill, dtype=dtype)
    else:
        a = ensure_channel(a, 0)

    return np.stack([r, g, b, a], axis=-1)
