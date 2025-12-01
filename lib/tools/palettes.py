"""

Palette helpers

"""
# import random
import numpy as np
from numpy.typing import NDArray
# from typing import Union, Sequence
from . import images
import matplotlib.colors as mcolors
# import colour  # pip install colour-science
# from matplotlib.colors import to_rgb
from collections.abc import Sequence

# __all__ = [
#     "soil_pallete",
#     "apply_palette",
#     "to_rgb",
#     "to_hex",
#     "save",
#     "load",
#     "render_gradient",
#     "random_gradient",
# ]

# soil pallete example (gradient strip)


def soil_pallete() -> NDArray[np.float32]:
    """
    example soil colors
    """
    soil_colors = np.array([
        to_rgb("#8b4513"),  # SaddleBrown (clay-rich soil)
        to_rgb("#f4a460"),  # SandyBrown (sand/loam)
        to_rgb("#d2b48c"),  # Tan (silt or light soil)
        to_rgb("#696969"),  # DimGray (shale/rock fragments)
        to_rgb("#deb887"),  # BurlyWood (weathered sandstone)
        to_rgb("#2f4f4f"),  # DarkSlateGray (basalt/organic-rich soil)
    ], dtype=np.float32)

    return soil_colors

# detect what type of pallete


ColorTuple = tuple[int, int, int]
GradientPoint = tuple[float, str | ColorTuple]


def detect_kind(
    palette: np.ndarray
    | Sequence[str]
    | Sequence[ColorTuple]
    | Sequence[GradientPoint]
) -> str:
    """
    Detect the type of palette or gradient representation.

    Parameters
    ----------
    palette : np.ndarray or list/tuple
        Palette candidate (hex strings, named colors, numeric array, gradient control points, etc.)

    Returns
    -------
    str
        One of: "hex_list", "named_list", "numeric_array", "image_strip",
                "numeric_list", "gradient", or "unknown"
    """
    if isinstance(palette, np.ndarray):
        if palette.ndim == 2 and palette.shape[1] == 3:
            return "numeric_array"
        elif palette.ndim == 3 and palette.shape[1] == 1 and palette.shape[2] == 3:
            return "image_strip"
        else:
            return "unknown_array"

    if isinstance(palette, (list, tuple)):
        # Gradient control points: sequence of (position, color)
        if all(
            isinstance(x, (list, tuple))
            and len(x) == 2
            and isinstance(x[0], (int, float))
            for x in palette
        ):
            return "gradient"

        if all(isinstance(x, str) and x.startswith("#") for x in palette):
            return "hex_list"
        elif all(isinstance(x, str) for x in palette):
            return "named_list"
        elif all(isinstance(x, (list, tuple, np.ndarray)) and len(x) == 3 for x in palette):
            return "numeric_list"
        else:
            return "unknown_list"

    return "unknown"

# apply gradient strip to heightmap


def apply_gradient_strip(
    heightmap: NDArray[np.float32],
    gradient: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Map a grayscale heightmap to a color image using a gradient strip.

    Parameters
    ----------
    heightmap : NDArray[np.float32]
        2D array of grayscale values (any range).
    gradient : NDArray[np.float32]
        Array of shape (N,3) with RGB colors in [0,1], e.g. from render_gradient().

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

    n_colors = len(gradient)

    # Scale heightmap into gradient index range
    scaled = hmap * (n_colors - 1)
    idx_low = np.floor(scaled).astype(np.int32)
    idx_high = np.clip(idx_low + 1, 0, n_colors - 1)
    t = scaled - idx_low

    # Linear interpolation between gradient entries
    colored = (1 - t)[..., None] * gradient[idx_low] + t[..., None] * gradient[idx_high]

    return np.clip(colored, 0, 1).astype(np.float32)

# turn any palette to gradient strip (N, 3)


def to_rgb(palette) -> NDArray[np.float32]:
    """
    Normalize any palette representation into (N,3) float array in [0,1].
    """
    kind = detect_kind(palette)

    if kind == "numeric_array":
        return palette.astype(np.float32)

    if kind == "image_strip":
        return palette.reshape((-1, 3)).astype(np.float32)

    if kind == "hex_list" or kind == "named_list":
        return np.vstack([mcolors.to_rgb(c) for c in palette]).astype(np.float32)

    if kind == "numeric_list":
        return np.vstack(palette).astype(np.float32)

    raise ValueError(f"Unsupported palette type: {kind}")

# turn any palette to hex list


def to_hex(palette) -> list[str]:
    """
    Normalize any palette representation into a list of hex strings '#RRGGBB'.
    """
    kind = detect_kind(palette)

    if kind == "hex_list":
        return palette

    if kind == "named_list":
        return [mcolors.to_hex(mcolors.to_rgb(c)) for c in palette]

    if kind == "numeric_array":
        return [mcolors.to_hex(c) for c in palette]

    if kind == "image_strip":
        arr = palette.reshape((-1, 3))
        return [mcolors.to_hex(c) for c in arr]

    if kind == "numeric_list":
        arr = np.vstack(palette)
        return [mcolors.to_hex(c) for c in arr]

    raise ValueError(f"Unsupported palette type: {kind}")

# save palette (as vertical color image)


def save(palette, filename: str) -> None:
    """
    Save a palette (autodetect) as standard palette image (vertical strip)
    (use a png filename ideally)
    """
    palette = to_rgb(palette)
    palette = palette.reshape((-1, 1, 3))
    images.save(palette, filename)

# load palette (as vertical color image)


def load(filename: str) -> NDArray[np.float32]:
    """
    Load a palette strip image (N,1,3) and collapse back to (N,3).
    """
    gradient = images.load_image(filename)
    return gradient.reshape((-1, 3))
