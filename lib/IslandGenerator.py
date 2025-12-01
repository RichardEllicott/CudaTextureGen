"""

generating random island shapes by warping

"""
import tools
import cuda_texture_gen
from numpy.typing import NDArray
import numpy as np


class IslandGenerator():

    _noise_generator = cuda_texture_gen.NoiseGenerator()
    _resample = cuda_texture_gen.Resample()

    # image dimensions
    width, height = 128, 128
    # starting circle diameter
    diameter = 40
    # Blur the starting circle
    pre_blur = 0.0

    # random seed
    seed = 0

    # optionally apply extra noise to the heightmap before warp
    height_noise_fade = 0.0  # [0, 1]
    height_noise_octaves = 8
    height_noise_period = 3

    # apply warp
    warp_octaves = 6
    warp_period = 3
    warp_strength = 0.25

    # blur result
    blur = 0.0

    @property
    def island(self) -> NDArray[np.float32]:
        return self.process()

    def process(self) -> NDArray[np.float32]:

        _island = tools.circle_array(self.width, self.height, self.diameter)

        # PRE BLUR
        if self.pre_blur > 0.0:
            _island = tools.blur_array(_island, self.pre_blur)

        # HEIGHT NOISE
        if self.height_noise_fade > 0.0:
            self.height_noise_fade = min(1.0, self.height_noise_fade)

            height_noise = tools.noise.fractal(self.width, self.height,
                                               octaves=self.height_noise_octaves,
                                               base_period=self.height_noise_period,
                                               seed=self.seed+1)
            product = _island * height_noise

            if self.height_noise_fade >= 1.0:
                _island = product
            else:
                _island = product * self.height_noise_fade + (1.0 - self.height_noise_fade) * _island

        # NOISE X
        noise_x = tools.noise.fractal(self.width, self.height,
                                      octaves=self.warp_octaves,
                                      base_period=self.warp_period,
                                      seed=self.seed+2)
        noise_x = noise_x * 2.0 - 1.0
        noise_x *= self.warp_strength

        # NOISE Y
        noise_y = tools.noise.fractal(self.width, self.height,
                                      octaves=self.warp_octaves,
                                      base_period=self.warp_period,
                                      seed=self.seed + 3)
        noise_y = noise_y * 2.0 - 1.0
        noise_y *= self.warp_strength

        # RESAMPLE
        self._resample.input = _island
        self._resample.map_x = noise_x
        self._resample.map_y = noise_y
        self._resample.process()
        _island = self._resample.output

        if self.blur > 0.0:
            _island = tools.blur_array(_island, self.blur)

        return _island


def main():

    for i in range(8):
        gen = IslandGenerator()
        gen.seed = i
        island = gen.island
        filename = f"./output/island_{i:02}.png"
        tools.images.save(island, filename)



if __name__ == "__main__":
    print("Running main logic...")
    main()  # This block runs only if executed directly
