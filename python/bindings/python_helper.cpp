#include "python_helper.h"

namespace python_helper {

// get numpy float array (note it's the same way as python with the height before width)
nb::ndarray<nb::numpy, float> get_numpy_float_array(int height, int width) {
    nb::module_ np = nb::module_::import_("numpy");                                       // import numpy
    nb::object arr_obj = np.attr("empty")(nb::make_tuple(height, width), "float32");      // create empty numpy array
    nb::ndarray<nb::numpy, float> arr = nb::cast<nb::ndarray<nb::numpy, float>>(arr_obj); // Cast to a typed ndarray
    return arr;
}

// get numpy float array (note it's the same way as python with the height before width)
nb::ndarray<nb::numpy, float> get_numpy_float_array(int height, int width, int depth) {
    nb::module_ np = nb::module_::import_("numpy");                                         // import numpy
    nb::object arr_obj = np.attr("empty")(nb::make_tuple(height, width, depth), "float32"); // create empty numpy array
    nb::ndarray<nb::numpy, float> arr = nb::cast<nb::ndarray<nb::numpy, float>>(arr_obj);   // Cast to a typed ndarray
    return arr;
}

// get nested python lists (example)
nb::object get_list_of_lists(int height, int width) {

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

void bind_python_helper(nb::module_ &m) {
    // EXAMPLE create a list of lists, we can't seem to make a numpy array though
    m.def("get_list_of_lists", [](int height, int width) {
        return get_list_of_lists(height, width);
    });

    m.def("get_2d_numpy_array", [](int height, int width) {
        return get_numpy_float_array(height, width);
    });
}

} // namespace python_helper