/*

numpy helper functions, to create arrays and convert them etc

*/
#pragma once

#include <cstring> // required for std::memcpy in linux (not windows)
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace python_helper {

namespace nb = nanobind; // shortcut

#pragma region TYPE_MAPPING

// Type-to-dtype mapping
template <typename T>
struct numpy_dtype; // no default

template <>
struct numpy_dtype<float> {
    static constexpr auto value = "float32";
};
template <>
struct numpy_dtype<double> {
    static constexpr auto value = "float64";
};
template <>
struct numpy_dtype<int> {
    static constexpr auto value = "int32";
};
template <>
struct numpy_dtype<long long> {
    static constexpr auto value = "int64";
};
template <>
struct numpy_dtype<uint8_t> {
    static constexpr auto value = "uint8";
};
template <>
struct numpy_dtype<bool> {
    static constexpr auto value = "bool_";
};
// …extend as needed

#pragma endregion

#pragma region CREATE_EMPTY_ARRAY

// get an uninitialized numpy array (1D)
template <typename T>
inline nb::ndarray<nb::numpy, T> get_numpy_array(int length) {
    nb::module_ np = nb::module_::import_("numpy");                     // import numpy
    nb::object array = np.attr("empty")(length, numpy_dtype<T>::value); // array = numpy.empty(length, dtype=T)
    return nb::cast<nb::ndarray<nb::numpy, T>>(array);                  // cast the nanobind object to a nanobind array
}

// get an uninitialized numpy array (2D)
template <typename T>
inline nb::ndarray<nb::numpy, T> get_numpy_array(int height, int width) {
    nb::module_ np = nb::module_::import_("numpy");                                            // import numpy
    nb::object array = np.attr("empty")(nb::make_tuple(height, width), numpy_dtype<T>::value); // array = numpy.empty((height, width), dtype=T)
    return nb::cast<nb::ndarray<nb::numpy, T>>(array);                                         // cast the nanobind object to a nanobind array
}

// get an uninitialized numpy array (3D)
template <typename T>
inline nb::ndarray<nb::numpy, T> get_numpy_array(int height, int width, int depth) {
    nb::module_ np = nb::module_::import_("numpy");                                                   // import numpy
    nb::object array = np.attr("empty")(nb::make_tuple(height, width, depth), numpy_dtype<T>::value); // array = numpy.empty((height, width, depth), dtype=T)
    return nb::cast<nb::ndarray<nb::numpy, T>>(array);                                                // cast the nanobind object to a nanobind array
}

#pragma endregion

#pragma region VECTOR

// Convert ndarray -> std::vector<T>
template <typename T>
inline std::vector<T> numpy_array_to_vector(nb::ndarray<T, nb::c_contig> arr) {
    if (arr.ndim() != 2)
        throw std::runtime_error("Input must be a 2D array");

    size_t size = arr.shape(0) * arr.shape(1);
    std::vector<T> result(size);
    std::memcpy(result.data(), arr.data(), size * sizeof(T));
    return result;
}

// Convert std::vector<T> -> ndarray<T>
template <typename T>
inline nb::ndarray<nb::numpy, T> vector_to_numpy_array(const std::vector<T> &source, int height, int width) {

    size_t size = static_cast<size_t>(height) * static_cast<size_t>(width);
    if (source.size() != size)
        throw std::runtime_error("Source vector size doesn't match requested dimensions");

    auto arr = get_numpy_array<T>(height, width); // uses the dtype traits

    std::memcpy(arr.data(), source.data(), size * sizeof(T));
    return arr;
}

#pragma endregion



#pragma region SHARED_MEMORY_NUMPY_EXPERIMENT

// // ⚠️ UNTESTED
// template <typename T>
// nb::ndarray<nb::numpy, T> array3d_to_numpy_view(core::types::Array3D<T> &source) {
//     return nb::ndarray<nb::numpy, T>(
//         source.data(),
//         {source.get_height(), source.get_width(), source.get_depth()},
//         {sizeof(T) * source.get_width() * source.get_depth(),
//          sizeof(T) * source.get_depth(),
//          sizeof(T)});
// }

#pragma endregion


} // namespace python_helper