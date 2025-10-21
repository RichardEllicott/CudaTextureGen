/*

python helper library

*/
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

// helpers to make python objects which is a bit convoluted from python
namespace python_helper {

namespace nb = nanobind; // shortcut

// create an empty 2D numpy array
nb::ndarray<nb::numpy, float> get_numpy_float_array(int height, int width);

// create an empty 3D numpy array
nb::ndarray<nb::numpy, float> get_numpy_float_array(int height, int width, int depth);

// 🚧 get a list of list (kept as an example)
nb::object get_list_of_lists(int h, int w);


// 🚧 this was just a test 🚧
void bind_python_helper(nb::module_ &m);

} // namespace python_helper