# add the build/python directory
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "build", "python"))


# Ubuntu Linux no longer wants to use PIP, but apt
# sudo apt install python3-pil
# sudo apt install python3-numpy

from PIL import Image
import numpy as np

# img = Image.open("test_blobs.png").convert("L")  # grayscale for simplicity
# arr = np.array(img, dtype=np.float32)         # shape (H, W)




import cuda_hello
from PIL import Image
import numpy as np

img = cuda_hello.generate_noise(256, 256, 0.02)

# Normalize to 0–255 and convert to uint8
norm = (img - img.min()) / (img.max() - img.min())
img_uint8 = (norm * 255).astype(np.uint8)

# Save as grayscale PNG
Image.fromarray(img_uint8, mode="L").save("terrain.png")