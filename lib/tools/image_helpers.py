"""
image helpers


# pip install imageio[ffmpeg]

"""
from typing import Any, Union, Sequence
from numpy.typing import NDArray
import numpy as np
import imageio.v2 as imageio  # version 2 loads to numpy arrays


def save_array_as_image(array: NDArray[Any], filename: Union[str, Sequence[str]]) -> None:
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


def load_image_as_array(filename: str, dtype: np.dtype = np.float32) -> NDArray[Any]:
    """
    Load an image file into a NumPy array.
    Preserves whatever dimensionality the image has:
    - Grayscale → 2D array (H, W)
    - RGB → 3D array (H, W, 3)
    - RGBA → 3D array (H, W, 4)
    - Float TIFFs → 2D or 3D depending on channels
    """
    array = imageio.imread(filename)
    return array.astype(dtype)


def save_frames_as_gif_animation(frames, filename, duration=0.1, loop=0) -> None:
    """
    Save a sequence of frames as an animated GIF.

    Parameters
    ----------
    filename : str
        Path to the output GIF file (e.g. "animation.gif").
    frames : list of numpy.ndarray
        List of image frames. Each frame is expected to be a NumPy array
        with values in the range [0, 1]. They will be scaled to [0, 255]
        and converted to 8-bit unsigned integers for GIF encoding.
    duration : float, optional
        Time between frames in seconds. Default is 0.1 (10 frames per second).
    loop : int, optional
        Number of times the GIF should loop. Default is 0 (infinite loop).

    Notes
    -----
    - Frames should be 2D (grayscale) or 3D (RGB) NumPy arrays.
    - Scaling assumes input frames are normalized floats in [0, 1].
    - Uses `imageio.mimsave` under the hood.
    """
    gif_frames = []

    # Convert each frame from [0,1] float to [0,255] uint8
    for frame in frames:
        gif_frame = (frame * 255).astype(np.uint8)
        gif_frames.append(gif_frame)

    # Save frames as an animated GIF
    imageio.mimsave(filename, gif_frames, format="GIF", duration=duration, loop=loop)


def save_frames_as_mp4(frames, filename, fps=10, codec='libx264'):
    """
    Save a sequence of frames as an MP4 video.

    Parameters
    ----------
    filename : str
        Path to the output video file (e.g. "animation.mp4").
    frames : list of numpy.ndarray
        List of image frames. Each frame should be a NumPy array
        with values in [0, 1] or [0, 255].
    fps : int, optional
        Frames per second. Default is 10.
    """
    # Convert frames to uint8 if needed
    video_frames = [(frame * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame for frame in frames]

    # Use imageio's writer for MP4
    with imageio.get_writer(filename, fps=fps, codec=codec) as writer:
        for frame in video_frames:
            writer.append_data(frame)
