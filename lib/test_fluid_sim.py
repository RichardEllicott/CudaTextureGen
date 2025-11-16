"""
testing 

still just blows up

https://copilot.microsoft.com/chats/ReYQfqECCvBRZbAT8yc9d


"""

from tools import *
import imageio
import os

folder = "output"

# region PRESETS


def PRESET_single_impulse(width=128, height=128):
    map = np.zeros((height, width), dtype=np.float32)
    cx, cy = width // 2, height // 2  # Set a single disturbance in the center
    map[cy, cx] = 1.0
    return map


def PRESET_gaussian_bump(width=128, height=128):

    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    cx, cy = width // 2, height // 2
    sigma = 10.0  # controls spread
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2)).astype(np.float32)


def PRESET_line_disturbance(width=128, height=128):

    map = np.zeros((height, width), dtype=np.float32)
    # Horizontal line across the middle
    map[height // 2, :] = 1.0

    return map


def PRESET_random_noise(width=128, height=128):
    return np.random.rand(height, width).astype(np.float32) * 0.1

# endregion


def make_gif(folder, filename_base="water_sim", steps=16, out_name="water_sim.gif"):
    frames = []
    for i in range(steps):
        fname = f"{folder}/{filename_base}{i:02}.png"
        frames.append(imageio.imread(fname))
    imageio.mimsave(out_name, frames, duration=0.1)  # duration = seconds per frame


def test(filename_base="water_sim", steps=32):

    save_to_gif = True
    save_to_pngs = False

    steps = 256

    width, height = 64, 64


    water_map = get_fractal_noise(width=width, height=height)
    # water_map = PRESET_single_impulse(width=width, height=height)
    # water_map = PRESET_gaussian_bump(width=width, height=height)
    # water_map = PRESET_line_disturbance(width=width, height=height)

    fluid_sim = cuda_texture_gen.FluidSimulation()
    fluid_sim.water_map = water_map
    

    fluid_sim.dt = 0.1
    fluid_sim.wave_speed = 5.0
    fluid_sim.cell_size = 1.0
    fluid_sim.damping = 0.01
    fluid_sim.steps = 1
    
    
    



    gif_frames = []

    for i in range(steps):
        fluid_sim.process()

        png_filename = f"{folder}/{filename_base}{i:02}.png"

        frame = fluid_sim.water_map

        # frame /= 2.0
        # frame += 0.5

        print(f"🎞️ {i:02}, min: {frame.min():.2f}, max: {frame.max():.2f}")

        if save_to_pngs:
            save_array_as_image(frame * 255, png_filename)

        if save_to_gif:
            # water_map is a 2D float32 array; scale to 0–255 and cast to uint8
            gif_frame = (frame * 255).astype(np.uint8)
            gif_frames.append(gif_frame)

    if save_to_gif:
        gif_filename = f"{folder}/{filename_base}.gif"
        imageio.mimsave(gif_filename, gif_frames, duration=0.1, loop=0)  # 0.1s per frame


test()
