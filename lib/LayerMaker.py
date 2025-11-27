"""

pip install colour-science

"""
import colour


import tools
import cuda_texture_gen
from numpy.typing import NDArray
from typing import Any, Optional
import numpy as np

from matplotlib.colors import to_rgb


class LayerMaker():

    input = None
    gradient = None

    def process():

        pass


def get_soil_pallete():

    soil_colors = np.array([
        to_rgb("#8b4513"),  # SaddleBrown (clay-rich soil)
        to_rgb("#f4a460"),  # SandyBrown (sand/loam)
        to_rgb("#d2b48c"),  # Tan (silt or light soil)
        to_rgb("#696969"),  # DimGray (shale/rock fragments)
        to_rgb("#deb887"),  # BurlyWood (weathered sandstone)
        to_rgb("#2f4f4f"),  # DarkSlateGray (basalt/organic-rich soil)
    ], dtype=np.float32)

    return soil_colors


def get_soil_pallete2():
    soil_colors = np.array([
        to_rgb("#8fb5f8"),
        to_rgb("#bed27b"),
        to_rgb("#f4a460"),
        to_rgb("#34783a"),
        to_rgb("#adcab1"),
    ], dtype=np.float32)

    return soil_colors


def perceptual_gradient(control_points, n_samples=256) -> np.ndarray:
    """
    Generate a gradient interpolated in Lab space using colour-science.
    """
    # Convert hex or RGB stops to Lab
    rgb_points = np.array([colour.utilities.as_float_array(colour.convert(c, 'sRGB', 'Lab'))
                           for c in control_points])

    # Interpolation positions
    stops = np.linspace(0, 1, len(rgb_points))
    query = np.linspace(0, 1, n_samples)

    # Interpolate each Lab channel
    lab_gradient = np.empty((n_samples, 3), dtype=np.float32)
    for i in range(3):
        lab_gradient[:, i] = np.interp(query, stops, rgb_points[:, i])

    # Convert back to sRGB
    rgb_gradient = colour.convert(lab_gradient, 'Lab', 'sRGB')
    return np.clip(rgb_gradient, 0, 1)


def get_perceptual_gradient():

    # Define control points (soil/rock tones)
    control_points = ["#8b4513", "#f4a460", "#d2b48c", "#696969", "#ffffff"]

    # Generate perceptual gradient
    rock_palette = perceptual_gradient(control_points, n_samples=256, space="Lab")

    return rock_palette

    # # Apply to heightmap (using your apply_palette function)
    # colored = apply_palette(heightmap, rock_palette, smooth=True)

    # Save result
    # cv2.imwrite("rock_gradient.png", (colored * 255).astype(np.uint8))


def main():

    from IslandGenerator import IslandGenerator

    island_generator = IslandGenerator()
    island = island_generator.island

    island = tools.blur_array(island, 12.0)

    tools.save_image(island, f"./output/layer_maker.height.png")

    # layer_maker = LayerMaker()
    # gradient = get_soil_pallete()
    gradient = get_perceptual_gradient()

    pallete_filename = f"./output/layer_maker.gradient.png"
    tools.save_palette(gradient, pallete_filename)
    pallete_filename = tools.load_palette(pallete_filename)

    color_map = tools.apply_palette(island, gradient, True)

    # tools.print_array_information(gradient)

    tools.save_image(color_map, f"./output/layer_maker.color_map.png")


if __name__ == "__main__":
    print("Running main logic...")
    main()  # This block runs only if executed directly
