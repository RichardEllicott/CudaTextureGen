"""

cuda tools

"""
import cuda_texture_gen
import numpy as np
from numpy.typing import NDArray


def normal_map(array: NDArray, strength: float = 1.0, wrap: bool = True):
    """
    generate normal map
    """
    return cuda_texture_gen.generate_normal_map(array, strength, wrap)


def ao_map(array: NDArray, radius: float = 1.0, wrap: bool = True, mode: int = 0):
    """
    generate ambient occlusion
    """
    return cuda_texture_gen.generate_ao_map(array, radius, wrap, mode)


def blur(input: NDArray, amount: float = 1.0, wrap: bool = True):
    """
    🐞 might not work if array gets copied inside (slight bug)
    """
    cuda_texture_gen.blur(input, amount, wrap)
