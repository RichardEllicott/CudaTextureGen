# add the build/python directory
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "build", "python"))


# Ubuntu Linux no longer wants to use PIP, but apt
# sudo apt install python3-pil
# sudo apt install python3-numpy

from PIL import Image
import numpy as np

img = Image.open("test_blobs.png").convert("L")  # grayscale for simplicity
arr = np.array(img, dtype=np.float32)         # shape (H, W)


# Erode
import cuda_hello


cuda_hello.blur(arr, 32.0, True) # modifies in place
Image.fromarray(arr.astype(np.uint8)).save("output/blurred.png")