"""
Image helpers with EXR support

Dependencies:
    pip install imageio[ffmpeg]
    pip install openexr
    pip install imath
"""

import numpy as np
import imageio
import imageio.v2 as imageio_v2  # version 2 loads to numpy arrays
import OpenEXR
import Imath
from pathlib import Path
from collections.abc import Sequence
from typing import cast
from numpy.typing import NDArray


ImageArray = NDArray[np.uint8] | NDArray[np.floating]

__all__ = [
    "save",
    "load",
]


def save(
    image_array: np.ndarray,
    filename: str | Sequence[str] | Path | Sequence[Path],
    verbose: bool = True,
    scale_pngs: bool = True
) -> None:
    """
    Save a NumPy array as an image.

    Supports:
      - .png  (uint8)
      - .tif/.tiff (float32)
      - .exr (float32, single or multi-channel)

    Args:
        image_array: NumPy array to save (2D for grayscale, 3D for RGB/RGBA)
        filename: Output filename(s). Can be a single path or sequence of paths.
        verbose: Print save messages
        scale_pngs: Scale float arrays to 0-255 range for PNG output
    """

    # Handle multiple filenames
    if isinstance(filename, (list, tuple)):
        for fname in filename:
            save(image_array, fname, verbose=verbose, scale_pngs=scale_pngs)
        return

    filename = cast(str, filename)
    ext = str(filename).lower()

    if verbose:
        print(f"💾 saving: {filename}")

    # -------------------------
    # PNG (8-bit)
    # -------------------------
    if ext.endswith(".png"):
        arr = image_array
        if scale_pngs and not np.issubdtype(arr.dtype, np.uint8):
            arr = arr * 255
        imageio.imwrite(filename, arr.astype(np.uint8))
        return

    # -------------------------
    # TIFF (float32)
    # -------------------------
    if ext.endswith((".tif", ".tiff")):
        imageio.imwrite(filename, image_array.astype(np.float32))
        return

    # -------------------------
    # EXR (float32)
    # -------------------------
    if ext.endswith(".exr"):
        arr = image_array.astype(np.float32)

        # Handle different array shapes
        if arr.ndim == 2:
            # Grayscale (H, W)
            h, w = arr.shape
            channels = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
            channel_data = {'Y': arr.tobytes()}

        elif arr.ndim == 3:
            h, w, c = arr.shape

            if c == 3:
                # RGB
                channels = {
                    'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
                channel_data = {
                    'R': arr[:, :, 0].tobytes(),
                    'G': arr[:, :, 1].tobytes(),
                    'B': arr[:, :, 2].tobytes()
                }
            elif c == 4:
                # RGBA
                channels = {
                    'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'A': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
                channel_data = {
                    'R': arr[:, :, 0].tobytes(),
                    'G': arr[:, :, 1].tobytes(),
                    'B': arr[:, :, 2].tobytes(),
                    'A': arr[:, :, 3].tobytes()
                }
            else:
                raise ValueError(f"EXR: unsupported channel count {c}. Expected 3 (RGB) or 4 (RGBA).")
        else:
            raise ValueError(f"EXR: array must be 2D or 3D, got {arr.ndim}D")

        header = OpenEXR.Header(w, h)
        header['channels'] = channels

        exr = OpenEXR.OutputFile(str(filename), header)
        exr.writePixels(channel_data)
        exr.close()
        return

    # -------------------------
    # Unsupported
    # -------------------------
    raise ValueError(f"Unsupported format: {filename}")


def load(
    filename: str | Path,
    dtype: np.dtype | type[np.generic] = np.float32,
    verbose: bool = True,
    scale_pngs: bool = True,
) -> np.ndarray:
    """
    Load an image file into a NumPy array.

    Preserves dimensionality:
    - Grayscale → 2D array (H, W)
    - RGB → 3D array (H, W, 3)
    - RGBA → 3D array (H, W, 4)
    - Float TIFFs → 2D or 3D depending on channels
    - EXR → single or multi-channel float32

    Args:
        filename: Path to image file
        dtype: Output data type
        verbose: Print load messages
        scale_pngs: Scale PNG uint8 values to 0-1 range

    Returns:
        NumPy array containing the image data
    """

    if verbose:
        print(f"💾 loading: {filename}")

    ext = str(filename).lower()

    # -------------------------
    # EXR (float32)
    # -------------------------
    if ext.endswith(".exr"):
        exr = OpenEXR.InputFile(str(filename))
        header = exr.header()

        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = header['channels'].keys()

        # Try to load RGB/RGBA first
        if 'R' in channels and 'G' in channels and 'B' in channels:
            r = np.frombuffer(exr.channel('R', pt), dtype=np.float32).reshape((height, width))
            g = np.frombuffer(exr.channel('G', pt), dtype=np.float32).reshape((height, width))
            b = np.frombuffer(exr.channel('B', pt), dtype=np.float32).reshape((height, width))

            if 'A' in channels:
                a = np.frombuffer(exr.channel('A', pt), dtype=np.float32).reshape((height, width))
                image_array = np.stack([r, g, b, a], axis=-1)
            else:
                image_array = np.stack([r, g, b], axis=-1)

        # Fall back to Y (grayscale)
        elif 'Y' in channels:
            raw = exr.channel('Y', pt)
            image_array = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
        else:
            raise ValueError(f"EXR: no supported channels found. Available channels: {list(channels)}")

        return image_array.astype(dtype)

    # -------------------------
    # PNG, TIFF, etc.
    # -------------------------
    image_array = imageio.imread(str(filename))

    if scale_pngs and ext.endswith(".png"):
        if np.issubdtype(image_array.dtype, np.uint8):
            image_array = image_array / 255.0

    return image_array.astype(dtype)


# Example usage
if __name__ == "__main__":
    # Create test data
    print("Creating test images...")

    # Grayscale test
    grayscale = np.random.rand(256, 256).astype(np.float32)
    save(grayscale, "test_grayscale.exr")
    save(grayscale, "test_grayscale.png")
    save(grayscale, "test_grayscale.tiff")

    # RGB test
    rgb = np.random.rand(256, 256, 3).astype(np.float32)
    save(rgb, "test_rgb.exr")
    save(rgb, "test_rgb.png")

    # Load back
    print("\nLoading images...")
    loaded_exr = load("test_grayscale.exr")
    loaded_png = load("test_grayscale.png")
    loaded_rgb_exr = load("test_rgb.exr")

    print(f"\nGrayscale EXR shape: {loaded_exr.shape}")
    print(f"Grayscale PNG shape: {loaded_png.shape}")
    print(f"RGB EXR shape: {loaded_rgb_exr.shape}")

    print("\n✅ All tests completed!")
