/*
create numpy arrays

get_array(length) // 1D
get_array(height, width) // 2D
get_array(height, width, depth) // 3D

*/
#pragma once

#include <cstring> // required for std::memcpy in linux (not windows)
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nanobind::helper::numpy {

namespace nb = nanobind; // shortcut

#pragma region UTILITY

// confirm if numpy array is c contiguous
// we only need this when we intend to modify numpy arrays in place
template <typename T>
bool is_c_contiguous(const nb::ndarray<T> &array) {
    size_t expected_stride = sizeof(T);

    for (int i = array.ndim() - 1; i >= 0; --i) {
        if (array.stride(i) != expected_stride) {
            return false;
        }
        expected_stride *= array.shape(i);
    }

    return true;
}

#pragma endregion

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
inline nb::ndarray<nb::numpy, T> get_array(int length) {
    nb::module_ np = nb::module_::import_("numpy");                     // import numpy
    nb::object array = np.attr("empty")(length, numpy_dtype<T>::value); // array = numpy.empty(length, dtype=T)
    return nb::cast<nb::ndarray<nb::numpy, T>>(array);                  // cast the nanobind object to a nanobind array
}

// get an uninitialized numpy array (2D)
template <typename T>
inline nb::ndarray<nb::numpy, T> get_array(int height, int width) {
    nb::module_ np = nb::module_::import_("numpy");                                            // import numpy
    nb::object array = np.attr("empty")(nb::make_tuple(height, width), numpy_dtype<T>::value); // array = numpy.empty((height, width), dtype=T)
    return nb::cast<nb::ndarray<nb::numpy, T>>(array);                                         // cast the nanobind object to a nanobind array
}

// get an uninitialized numpy array (3D)
template <typename T>
inline nb::ndarray<nb::numpy, T> get_array(int height, int width, int depth) {
    nb::module_ np = nb::module_::import_("numpy");                                                   // import numpy
    nb::object array = np.attr("empty")(nb::make_tuple(height, width, depth), numpy_dtype<T>::value); // array = numpy.empty((height, width, depth), dtype=T)
    return nb::cast<nb::ndarray<nb::numpy, T>>(array);                                                // cast the nanobind object to a nanobind array
}

#pragma endregion

#pragma region N_DIMENSION_SUPPORT


// get an uninitialized numpy array with arbitrary shape
template <typename T, int Dim>
inline nb::ndarray<nb::numpy, T> get_array(const std::array<size_t, Dim>& shape) {
    nb::module_ np = nb::module_::import_("numpy");

    nb::object array;
    if constexpr (Dim == 1) {
        array = np.attr("empty")(nb::make_tuple(shape[0]), numpy_dtype<T>::value);
    } else if constexpr (Dim == 2) {
        array = np.attr("empty")(nb::make_tuple(shape[0], shape[1]), numpy_dtype<T>::value);
    } else if constexpr (Dim == 3) {
        array = np.attr("empty")(nb::make_tuple(shape[0], shape[1], shape[2]), numpy_dtype<T>::value);
    } else {
        // Fallback for higher dimensions: build list -> convert to tuple
        nb::list py_shape;
        for (int i = 0; i < Dim; ++i)
            py_shape.append(nb::int_(shape[i]));
        array = np.attr("empty")(nb::tuple(py_shape), numpy_dtype<T>::value);
    }

    return nb::cast<nb::ndarray<nb::numpy, T>>(array);
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

} // namespace nanobind::helper::numpy