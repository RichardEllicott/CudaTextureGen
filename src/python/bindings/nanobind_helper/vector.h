
/*
convert numpy arrays to and from c++ vectors
*/
#pragma once

#include "numpy.h"
#include <cstring> // required for std::memcpy in linux (not windows)
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nanobind::helper::numpy {

namespace nb = nanobind;

#pragma region VECTOR

// Convert ndarray -> std::vector<T>
template <typename T>
inline std::vector<T> array_to_vector(nb::ndarray<T, nb::c_contig> array) {
    if (array.ndim() != 2)
        throw std::runtime_error("Input must be a 2D array");

    size_t size = array.shape(0) * array.shape(1);
    std::vector<T> result(size);
    std::memcpy(result.data(), array.data(), size * sizeof(T));
    return result;
}

// Convert std::vector<T> -> ndarray<T>
template <typename T>
inline nb::ndarray<nb::numpy, T> vector_to_array(const std::vector<T> &source, int height, int width) {

    size_t size = static_cast<size_t>(height) * static_cast<size_t>(width);
    if (source.size() != size)
        throw std::runtime_error("Source vector size doesn't match requested dimensions");

    auto array = get_numpy_array<T>(height, width); // uses the dtype traits

    std::memcpy(array.data(), source.data(), size * sizeof(T));
    return array;
}

#pragma endregion

} // namespace python_helper