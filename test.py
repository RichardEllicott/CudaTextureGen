# add the build/python directory
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "python"))


# Ubuntu Linux no longer wants to use PIP, but apt
# sudo apt install python3-pil
# sudo apt install python3-numpy

from PIL import Image
import numpy as np

# img = Image.open("perlin_noise_example.png").convert("L")  # grayscale for simplicity
img = Image.new(mode="RGB", size=(128, 128))
arr = np.array(img, dtype=np.float32)         # shape (H, W)



# print(type(arr))

# Erode
import cuda_hello
# cuda_hello.erosion(arr, steps=50)   # modifies in place
# Image.fromarray(arr.astype(np.uint8)).save("output/eroded.png")


# ret = cuda_hello.generate_noise(256, 256, 2)

# print(ret)


# ret = cuda_hello.make_array(256, 256);


# ret = cuda_hello.make_list(16, 16)s
# print(ret)


ret = cuda_hello.make_array_efficient(16, 16)
print(ret)