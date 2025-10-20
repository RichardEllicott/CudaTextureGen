/*

*/
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>



inline void test_find_this_lib(){



}


// helpers to make python objects which is a bit convoluted from python
namespace python_helper {

namespace nb = nanobind; // shortcut

// create an empty 2D numpy array
inline nb::ndarray<nb::numpy, float> get_numpy_float_array(int height, int width) {
    nb::module_ np = nb::module_::import_("numpy");                                       // import numpy
    nb::object arr_obj = np.attr("empty")(nb::make_tuple(height, width), "float32");      // create empty numpy array
    nb::ndarray<nb::numpy, float> arr = nb::cast<nb::ndarray<nb::numpy, float>>(arr_obj); // Cast to a typed ndarray
    return arr;
}

// create an empty 3D numpy array
inline nb::ndarray<nb::numpy, float> get_numpy_float_array(int height, int width, int depth) {
    nb::module_ np = nb::module_::import_("numpy");                                         // import numpy
    nb::object arr_obj = np.attr("empty")(nb::make_tuple(height, width, depth), "float32"); // create empty numpy array
    nb::ndarray<nb::numpy, float> arr = nb::cast<nb::ndarray<nb::numpy, float>>(arr_obj);   // Cast to a typed ndarray
    return arr;
}

inline nb::object get_list_of_lists(int h, int w) {

    std::vector<nb::list> rows;
    rows.reserve(h);

    for (int y = 0; y < h; ++y) {
        nb::list row;
        for (int x = 0; x < w; ++x) {
            row.append(0.0f); // or some value
        }
        rows.push_back(row);
    }

    nb::list outer;
    for (auto &r : rows)
        outer.append(r);

    return outer; // Python will see a list of lists
}

inline static void bind_helpers(nb::module_ &m) {
    // EXAMPLE create a list of lists, we can't seem to make a numpy array though
    m.def("get_list_of_lists", [](int h, int w) {
        return get_list_of_lists(h, w);
    });

    m.def("get_2d_numpy_array", [](int h, int w) {
        return get_numpy_float_array(h, w);
    });
}

} // namespace python_helper