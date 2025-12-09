"""

Palette helpers

"""
import colorsys
import random
import numpy as np
from . import images

# from numpy.typing import NDArray
# from typing import Union, Sequence
# import matplotlib.colors as mcolors
# import colour  # pip install colour-science
# from matplotlib.colors import to_rgb


class Gradient:
    """
    Gradient object that uses a (N, 4) array
    (position, r, g, b)
    """

    def __init__(self):
        # Start with an empty (0,4) array of float32
        self._data = np.zeros((0, 4), dtype=np.float32)

    def _hex_to_rgb(self, hex_str: str) -> tuple[float, float, float]:
        """
        Convert #RRGGBB hex string to normalized RGB tuple in [0,1].

        Raises
        ------
        ValueError
            If the string is not a valid 7-character hex color (#RRGGBB).
        """
        if not isinstance(hex_str, str):
            raise TypeError("hex_str must be a string")

        hex_str = hex_str.lstrip('#')
        if len(hex_str) != 6 or any(c not in "0123456789abcdefABCDEF" for c in hex_str):
            raise ValueError(f"Invalid hex color string: {hex_str!r}")

        return (
            int(hex_str[0:2], 16) / 255.0,
            int(hex_str[2:4], 16) / 255.0,
            int(hex_str[4:6], 16) / 255.0,
        )

    def add_control_point(self, position: float, color):
        """
        Add a control point (position, color) to the gradient.
        """
        if isinstance(color, str) and color.startswith('#'):
            rgb = self._hex_to_rgb(color)
        elif isinstance(color, (tuple, list)) and len(color) == 3:
            # Assume already normalized floats in [0,1]
            rgb = tuple(float(c) for c in color)
        else:
            raise ValueError("Color must be hex '#RRGGBB' or tuple/list of 3 floats")

        # Always append a new row
        new_row = np.array([[position, *rgb]], dtype=np.float32)
        self._data = np.vstack([self._data, new_row])

    def sort(self) -> None:
        """
        Sort control points by position (first column).
        """
        if len(self._data) > 0:
            indices = np.argsort(self._data[:, 0])
            self._data = self._data[indices]

    def normalize(self):
        """
        Rescale positions so they span [0,1].
        """
        if len(self._data) == 0:
            return
        positions = self._data[:, 0]
        min_pos, max_pos = positions.min(), positions.max()
        if max_pos > min_pos:
            self._data[:, 0] = (positions - min_pos) / (max_pos - min_pos)

    def render(self, n_samples: int = 256) -> np.ndarray:
        """
        Render gradient as (n_samples, 3) RGB array.
        """
        if len(self._data) == 0:
            return np.zeros((n_samples, 3), dtype=np.float32)

        positions = self._data[:, 0]
        colors = self._data[:, 1:4]

        query = np.linspace(0, 1, n_samples)

        gradient = np.empty((n_samples, 3), dtype=np.float32)
        for i in range(3):
            gradient[:, i] = np.interp(query, positions, colors[:, i])

        return gradient

    def get_data(self) -> np.ndarray:
        """
        Return the full (N,4) array of control points.
        """
        return self._data

    def set_data(self, data: np.ndarray):
        """
        Set the control points array, validating shape and dtype.
        data shape must be (N,4)
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a NumPy ndarray.")
        if data.dtype != np.float32:
            raise TypeError("Data must have dtype np.float32.")
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError("Data must have shape (N,4).")

        self._data = data

    def save(self, filename):
        """
        save to image, ideally use a tif to preserve 32 bit floats
        """
        images.save(self._data, filename)

    def load(self, filename):
        """
        load from image, ideally use a tif to preserve 32 bit floats
        """
        self.set_data(images.load(filename))


# hex stops example
gradient_example = [
    (0.0, "#8b4513"),  # start
    (0.5, "#f4a460"),  # midpoint
    (1.0, "#ffffff")   # end
]

# generate random gradient as tuples


soil_pallete_01 = [
    "#8b4513",  # SaddleBrown (clay-rich soil)
    "#f4a460",  # SandyBrown (sand/loam)
    "#d2b48c",  # Tan (silt or light soil)
    "#696969",  # DimGray (shale/rock fragments)
    "#deb887",  # BurlyWood (weathered sandstone)
    "#2f4f4f",  # DarkSlateGray (basalt/organic-rich soil)
]


def get_test_gradient_01(seed: int = 0) -> Gradient:
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    gradient = Gradient()

    for color in soil_pallete_01:
        gradient.add_control_point(random.random(), color)

    # Optional: sort by position so gradient flows left→right
    gradient.sort()
    gradient.normalize()

    return gradient


def random_gradient_from_pallete(pallete: list[str], seed: int = 0, n_points: int = 4):
    """
    Create a test gradient with a fixed seed and a chosen number of control points.
    Colors are picked randomly from soil_pallete_01, repeats allowed.
    """
    random.seed(seed)
    np.random.seed(seed)

    gradient = Gradient()

    for _ in range(n_points):
        color = random.choice(pallete)   # allows repeats
        position = random.random()               # random float in [0,1]
        gradient.add_control_point(position, color)

    # Sort and normalize positions
    gradient.sort()
    gradient.normalize()

    return gradient


soil_pallete_02 = [
    "#8b4513",  # SaddleBrown (clay-rich soil)
    "#939c2e",  # SaddleBrown (clay-rich soil)
    "#d2b48c",  # Tan (silt or light soil)
    "#deb887",  # BurlyWood (weathered sandstone)
    "#ba8c4f",  # BurlyWood (weathered sandstone)
]


def random_palette(center_hue: float,
                   n: int = 5,
                   hue_std: float = 10.0,
                   sat_range=(0.3, 0.6),
                   val_range=(0.3, 0.7),
                   seed: int | None = None) -> list[str]:
    """
    Generate a palette of hex color strings clustered around a center hue.
    Procedural: reproducible with a seed.

    Args:
        center_hue: Hue in degrees [0, 360).
        n: Number of colors.
        hue_std: Standard deviation for hue jitter (degrees).
        sat_range: Min/max saturation range.
        val_range: Min/max value (brightness) range.
        seed: Optional integer seed for reproducibility.

    Returns:
        List of hex strings like '#aabbcc'.
    """
    if seed is not None:
        random.seed(seed)

    colors = []
    for _ in range(n):
        hue = random.gauss(center_hue, hue_std) % 360
        sat = random.uniform(*sat_range)
        val = random.uniform(*val_range)

        r, g, b = colorsys.hsv_to_rgb(hue/360.0, sat, val)
        r, g, b = int(r*255), int(g*255), int(b*255)

        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def get_test_gradient_02(seed: int = 0, n_points: int = 32) -> Gradient:
    # """
    # Create a test gradient with a fixed seed and a chosen number of control points.
    # Colors are picked randomly from soil_pallete_01, repeats allowed.
    # """

    # return random_gradient_from_pallete(soil_pallete_01, seed, n_points)

    palette1 = random_palette(center_hue=30, n=n_points, seed=42)
    print(palette1)
    return random_gradient_from_pallete(palette1, seed, n_points)
