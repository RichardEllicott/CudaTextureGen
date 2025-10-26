#include "python_helper.h"

namespace python_helper {

nb::ndarray<nb::numpy, float> get_numpy_float_array(int height, int width) {
    nb::module_ np = nb::module_::import_("numpy");                                       // import numpy
    nb::object arr_obj = np.attr("empty")(nb::make_tuple(height, width), "float32");      // create empty numpy array
    nb::ndarray<nb::numpy, float> arr = nb::cast<nb::ndarray<nb::numpy, float>>(arr_obj); // Cast to a typed ndarray
    return arr;
}

nb::ndarray<nb::numpy, float> get_numpy_float_array(int height, int width, int depth) {
    nb::module_ np = nb::module_::import_("numpy");                                         // import numpy
    nb::object arr_obj = np.attr("empty")(nb::make_tuple(height, width, depth), "float32"); // create empty numpy array
    nb::ndarray<nb::numpy, float> arr = nb::cast<nb::ndarray<nb::numpy, float>>(arr_obj);   // Cast to a typed ndarray
    return arr;
}

std::vector<float> numpy_array_to_vector(nb::ndarray<float, nb::c_contig> arr) {
    if (arr.ndim() != 2)
        throw std::runtime_error("Input must be a 2D float32 array");

    size_t size = arr.shape(0) * arr.shape(1);
    std::vector<float> result(size);
    std::memcpy(result.data(), arr.data(), size * sizeof(float));
    return result;
}

nb::ndarray<nb::numpy, float> vector_to_numpy_array(const std::vector<float> &source, int height, int width) {

    size_t size = width * height;
    if (source.size() != size)
        throw std::runtime_error("Source vector size doesn't match requested dimensions");

    auto arr = get_numpy_float_array(height, width);

    std::memcpy(arr.data(), source.data(), size * sizeof(float)); // copy the memory
    return arr;
}

core::Array2D<float> numpy_array_to_array2d(nb::ndarray<float, nb::c_contig> arr) {
    if (arr.ndim() != 2)
        throw std::runtime_error("Input must be a 2D float32 array");

    size_t height = arr.shape(0);
    size_t width = arr.shape(1);

    core::Array2D<float> result(width, height);
    std::memcpy(result.data(), arr.data(), width * height * sizeof(float));
    return result;
}

nb::ndarray<nb::numpy, float> array2d_to_numpy_array(const core::Array2D<float> &source) {
    size_t width = source.get_width();
    size_t height = source.get_height();
    size_t size = width * height;

    auto arr = get_numpy_float_array(height, width); // assumes row-major layout

    std::memcpy(arr.data(), source.data(), size * sizeof(float)); // copy the memory
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