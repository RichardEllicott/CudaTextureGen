/*
convert numpy arrays to and from Array2D and Array3D's
*/
#pragma once

#include "core/types/array_2d.h"
#include "core/types/array_3d.h"
#include "numpy.h"

#include <cstring> // required for std::memcpy in linux (not windows)
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nanobind::helper {

namespace nb = nanobind; // shortcut

#pragma region ARRAY2D

// ndarray<T> -> core::types::Array2D<T>
template <typename T>
inline core::types::Array2D<T> numpy_array_to_array2d(nb::ndarray<T, nb::c_contig> arr) {
    if (arr.ndim() != 2)
        throw std::runtime_error("Input must be a 2D array");

    size_t height = arr.shape(0);
    size_t width = arr.shape(1);

    core::types::Array2D<T> result(width, height);
    std::memcpy(result.data(), arr.data(), width * height * sizeof(T));
    return result;
}

// core::types::Array2D<T> -> ndarray<T>
template <typename T>
inline nb::ndarray<nb::numpy, T> array2d_to_numpy_array(const core::types::Array2D<T> &source) {
    size_t width = source.get_width();
    size_t height = source.get_height();
    size_t size = width * height;

    auto arr = nb::helper::numpy::get_array<T>(height, width); // uses dtype traits

    std::memcpy(arr.data(), source.data(), size * sizeof(T));
    return arr;
}

#pragma endregion

#pragma region ARRAY3D

// ndarray<T> -> core::types::Array3D<T>
template <typename T>
inline core::types::Array3D<T> numpy_array_to_array3d(nb::ndarray<T, nb::c_contig> arr) {
    if (arr.ndim() != 3)
        throw std::runtime_error("Input must be a 3D array");

    size_t height = arr.shape(0);
    size_t width = arr.shape(1);
    size_t depth = arr.shape(2);

    core::types::Array3D<T> result(width, height, depth);
    std::memcpy(result.data(), arr.data(), width * height * depth * sizeof(T));
    return result;
}

// core::types::Array3D<T> -> ndarray<T>
template <typename T>
inline nb::ndarray<nb::numpy, T> array3d_to_numpy_array(const core::types::Array3D<T> &source) {
    size_t width = source.get_width();
    size_t height = source.get_height();
    size_t depth = source.get_depth();
    size_t size = width * height * depth;

    auto arr = nb::helper::numpy::get_array<T>(height, width, depth); // overload for 3D

    std::memcpy(arr.data(), source.data(), size * sizeof(T));
    return arr;
}

#pragma endregion

}