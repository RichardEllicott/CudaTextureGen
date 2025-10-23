"""

"""

import builtins  # snake print
import os
import platform
import sys
# import python_bootstrap # bootstrap to our fresh compiled module
import inspect  # function reflect
from PIL import Image
import numpy as np
from pathlib import Path
import cuda_texture_gen
#


def test_cuda_hello():
    print('{}()...'.format(inspect.currentframe().f_code.co_name))
    cuda_texture_gen.cuda_hello()


test_cuda_hello()