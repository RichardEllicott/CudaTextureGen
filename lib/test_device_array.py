"""

Erosion7 is the same as Erosion5 but has a new device array pattern

this makes it easier and quicker to upload and download data to the gpu and to therefore make animations

"""
from matplotlib.colors import to_rgb
import tools
import cuda_texture_gen
import numpy as np


import numpy as np
import cuda_texture_gen


def test():
    # Create your device array
    device_array = cuda_texture_gen.DeviceArrayFloat2D()

    # Make a simple pattern so you can visually confirm correctness
    h, w = 4, 4
    host = np.arange(h * w, dtype=np.int32).reshape(h, w)
    # host = np.array(["a", "bb"], dtype="S2")          # Raw bytes, fixed width

    print("Host array:")
    print(host)

    # Upload to device
    device_array.array = host

    # Download back
    out = device_array.array

    print("\nDownloaded array:")
    print(out)

    # Check equality
    print("\nMatch:", np.allclose(host, out))

    print(f"device_array.dev_ptr() = {device_array.dev_ptr()}")
    print(f"device_array.size() = {device_array.size()}")

    graph_node = cuda_texture_gen.GraphNode()
    print(f"graph_node.output = {graph_node.output}")
    graph_node.output = device_array
    print(f"graph_node.output = {graph_node.output}")


# test()

def connect_node(
        from_node: cuda_texture_gen.GraphNode,
        from_port: int,
        to_node: cuda_texture_gen.GraphNode,
        to_port: int):

    to_node.connect_input(from_node, from_port, to_port)


def test2():

    graph_node1 = cuda_texture_gen.GraphNode()
    graph_node2 = cuda_texture_gen.GraphNode()

    # print(graph_node1.connect_output(0, graph_node2, 0))
    # print(graph_node2.connect_input(graph_node1, 0, 0))

    connect_node(graph_node1, 0, graph_node2, 0)
    


# test2()


def test_gui():
    import tkinter as tk

    root = tk.Tk()
    root.title("My GUI")
    root.geometry("400x300")

    label = tk.Label(root, text="Hello world")
    label.pack()

    root.mainloop()



test_gui()