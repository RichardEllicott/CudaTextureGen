/*

python helper library

functions to covert numpy arrays back and forth to std::vector and core::Array2D

*/
#pragma once

#include "python_helper/array_nd.h"
#include "python_helper/device_array.h"
#include "python_helper/numpy_array.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

// helpers to make python objects which is a bit convoluted from python
namespace python_helper {

namespace nb = nanobind; // shortcut

// get nested python lists (example)
inline nb::object get_list_of_lists(int height, int width) {

    std::vector<nb::list> rows;
    rows.reserve(height);

    for (int y = 0; y < height; ++y) {
        nb::list row;
        for (int x = 0; x < width; ++x) {
            row.append(0.0f); // or some value
        }
        rows.push_back(row);
    }

    nb::list outer;
    for (auto &r : rows)
        outer.append(r);

    return outer; // Python will see a list of lists
}

} // namespace python_helper