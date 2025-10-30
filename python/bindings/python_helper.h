/*

python helper library

functions to covert numpy arrays back and forth to std::vector and core::Array2D

*/
#pragma once

#include "core.h"
#include "python_helper/numpy_array_helper.h"
#include <cstring> // required for std::memcpy in linux (not windows)
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

// helpers to make python objects which is a bit convoluted from python
namespace python_helper {

namespace nb = nanobind; // shortcut

// // create an empty 2D numpy array
// nb::ndarray<nb::numpy, float> get_numpy_float_array(int height, int width);

// // create an empty 3D numpy array
// nb::ndarray<nb::numpy, float> get_numpy_float_array(int height, int width, int depth);

// // copy from a numpy array (2D) to a std::vector
// std::vector<float> numpy_array_to_vector(nb::ndarray<float, nb::c_contig> arr);

// // copy a std::vector to a numpy array (2D)
// nb::ndarray<nb::numpy, float> vector_to_numpy_array(const std::vector<float> &source, int height, int width);

// // copy from a numpy 2D array to a new core::Array2D (custom type)
// core::Array2D<float> numpy_array_to_array2d(nb::ndarray<float, nb::c_contig> arr);

// // copy from core::Array2D to a numpy array
// nb::ndarray<nb::numpy, float> array2d_to_numpy_array(const core::Array2D<float> &source);

//
//
//
//
//

// 🚧 get a list of list (kept as an example)
nb::object get_list_of_lists(int h, int w);

// 🚧 this was just a test 🚧
void bind_python_helper(nb::module_ &m);

//
//
//
//

} // namespace python_helper