"""

basic test, no numpy required

"""
import python_bootstrap  # bootstrap to our fresh compiled module
import cuda_texture_gen

print("🐍 Basic Python Test...")
print(dir(cuda_texture_gen))


# print(cuda_texture_gen.cuda_hello())
print(cuda_texture_gen.test())

