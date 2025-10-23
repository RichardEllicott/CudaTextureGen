"""
Wrapper to load cuda_texture_gen from the fresh compile (without installing it)

 # noqa: E402
 is the only way to stop the python format from screwing up the order

"""
import sys  # noqa: E402
import platform  # noqa: E402
import os  # noqa: E402
root_dir = os.path.dirname(__file__)  # noqa: E402
if platform.system() == "Windows":  # noqa: E402
    build_subdir = os.path.join("..", "build", "windows")  # noqa: E402
else:  # noqa: E402
    build_subdir = os.path.join("..", "build", "linux")  # noqa: E402
sys.path.append(os.path.join(root_dir, build_subdir, "python"))  # noqa: E402
# import cuda_hello  # noqa: E402
import cuda_texture_gen  # noqa: E402
