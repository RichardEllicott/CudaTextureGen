# add the build/python directory
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "build", "python"))

# alternative could be running this command instead (also adds build/python)
# PYTHONPATH=build/python python3 test.py


import cuda_hello


print(cuda_hello.hello())   # "Hello from Python!"
cuda_hello.cuda_hello()     # prints "Hello from CPU" + GPU thread messages
