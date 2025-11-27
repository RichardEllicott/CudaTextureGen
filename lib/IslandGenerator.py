"""

generating random island shapes by warping

"""
import tools
import cuda_texture_gen
from numpy.typing import NDArray
from typing import Any, Optional
import numpy as np


class IslandGenerator():

    warp_strength = 0.25

    seed = 0

    width, height = 128, 128
    diameter = 40

    _noise_generator = cuda_texture_gen.NoiseGenerator()
    _resample = cuda_texture_gen.Resample()

    blur = 0.0

    @property
    def island(self) -> NDArray[np.float32]:
        return self.process()

    def process(self) -> NDArray[np.float32]:

        island = tools.circle_array(self.width, self.height, self.diameter)

        # self.noise_generator.period = 3
        # self.noise_generator.seed = self.seed

        # noise1 = self.noise_generator.generate(self.width, self.height)
        noise1 = tools.fractal_noise(self.width, self.height, octaves=6, base_period=3, seed=self.seed)
        noise1 = noise1 * 2.0 - 1.0
        noise1 *= self.warp_strength

        # self.noise_generator.seed += 1

        # noise2 = self.noise_generator.generate(self.width, self.height)
        noise2 = tools.fractal_noise(self.width, self.height, octaves=6, base_period=3, seed=self.seed + 1)
        noise2 = noise1 * 2.0 - 1.0
        noise2 *= self.warp_strength

        self._resample.input = island

        self._resample.map_x = noise1
        self._resample.map_y = noise2
        self._resample.process()

        island = self._resample.output

        if self.blur > 0.0:
            # tools.print_array_information(island)
            # island = cuda_texture_gen.blur(island, 1.0)
            island = tools.blur_array(island, self.blur)

        return island


def main():

    for i in range(8):
        gen = IslandGenerator()
        gen.seed = i
        island = gen.island
        filename = f"./output/island_{i:02}.png"
        tools.save_array_as_image(island * 255, filename)


if __name__ == "__main__":
    print("Running main logic...")
    main()  # This block runs only if executed directly
