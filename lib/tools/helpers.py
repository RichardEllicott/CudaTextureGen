"""

standard tools interface, load our library and hook up to it



"""
import numpy as np
import inspect
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from . import arrays


# def print_current_function():
#     """
#     print function call
#     """
#     frame = inspect.currentframe().f_back  # caller's frame
#     func_name = frame.f_code.co_name
#     args, _, _, values = inspect.getargvalues(frame)
#     arg_str = ', '.join(f"{arg}={values[arg]!r}" for arg in args)
#     print(f"{func_name}({arg_str})...")


from types import FrameType


def print_current_function() -> None:
    """
    Print the caller's function name and arguments.
    """
    frame: FrameType | None = inspect.currentframe()
    if frame is None or frame.f_back is None:
        print("<no caller>")
        return

    caller: FrameType = frame.f_back
    func_name: str = caller.f_code.co_name
    args, _, _, values = inspect.getargvalues(caller)

    arg_str: str = ', '.join(f"{arg}={values[arg]!r}" for arg in args)
    print(f"{func_name}({arg_str})...")


# # Split into RGB channels
# R = arr[:, :, 0]  # Red channel
# G = arr[:, :, 1]  # Green channel
# B = arr[:, :, 2]  # Blue channel


def smooth_layered_gradient(height_map, band_colors=[
        to_rgb("#8fb5f8"),
        to_rgb("#bed27b"),
        to_rgb("#f4a460"),
        to_rgb("#34783a"),
        to_rgb("#adcab1"),
    ]
):
    """
    height_map: 2D array normalized to [0, 1]
    band_colors: list of RGB tuples (e.g., from to_rgb)
    """
    bands = np.linspace(0, 1, len(band_colors))
    rgb_map = np.zeros((*height_map.shape, 3), dtype=np.float32)

    for i in range(len(bands) - 1):
        lower, upper = bands[i], bands[i + 1]
        mask = (height_map >= lower) & (height_map <= upper)
        t = (height_map[mask] - lower) / (upper - lower)  # blend factor [0,1]

        c0 = np.array(band_colors[i])
        c1 = np.array(band_colors[i + 1])
        rgb_map[mask] = (1 - t)[:, None] * c0 + t[:, None] * c1

    return rgb_map


def apply_color_map(height_map, cmap="terrain"):
    """
    apply matplotlib colormap

    Choose a colormap (e.g., 'terrain', 'viridis', 'plasma', or define your own)
    """
    colormap = plt.get_cmap(cmap)
    return colormap(height_map)
