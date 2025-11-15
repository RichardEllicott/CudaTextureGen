/*

seperating out

*/
#pragma once

#include "cuda_types.cuh"
#include "numpy_array.h" // numpy helper in same folder
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <cstring> // required for std::memcpy in linux (not windows)

namespace python_helper {

namespace nb = nanobind; // shortcut

#pragma region DEVICE_ARRAY_1D

// download a DeviceArray's data to a numpy array
template <typename T>
inline nb::ndarray<nb::numpy, T> device_array_to_numpy(const core::cuda::DeviceArray<T> &device_array) {
    auto array = python_helper::get_numpy_array<T>(device_array.size()); // create numpy array
    device_array.download(array.data());                                 // download the data into the numpy array
    return array;
}

// upload a numpy array's data to a DeviceArray
template <typename T>
inline void numpy_to_device_array(const nb::ndarray<T, nb::c_contig> &array, core::cuda::DeviceArray<T> &device_array) {
    if (array.ndim() != 1) {
        throw std::runtime_error("Input must be a 1D NumPy array");
    }
    device_array.upload(array.data(), array.shape(0)); // Upload raw pointer data into device array
}

#pragma endregion

#pragma region DEVICE_ARRAY_2D

// download a DeviceArray2D's data to a numpy array
template <typename T>
inline nb::ndarray<nb::numpy, T> device_array_2d_to_numpy(const core::cuda::DeviceArray2D<T> &device_array_2d) {
    auto array = python_helper::get_numpy_array<T>(device_array_2d.height(), device_array_2d.width()); // create numpy array
    device_array_2d.download(array.data());                                                            // download the data into the numpy array
    return array;
}

// upload a numpy array's data to a DeviceArray2D
template <typename T>
inline void numpy_to_device_array_2d(const nb::ndarray<T, nb::c_contig> &array, core::cuda::DeviceArray2D<T> &device_array_2d) {
    if (array.ndim() != 2) {
        throw std::runtime_error("Input must be a 2D NumPy array");
    }
    device_array_2d.upload(array.data(), array.shape(1), array.shape(0)); // Upload raw pointer data into device array
}

#pragma endregion

#pragma DEVICE_ARRAY_3D

// download a DeviceArray3D's data to a numpy array
template <typename T>
inline nb::ndarray<nb::numpy, T> device_array_3d_to_numpy(const core::cuda::DeviceArray3D<T> &device_array_3d) {
    auto array = python_helper::get_numpy_array<T>(device_array_3d.height(), device_array_3d.width(), device_array_3d.depth()); // create numpy array
    device_array_3d.download(array.data());                                                                                     // download the data into the numpy array
    return array;
}

// upload a numpy array's data to a DeviceArray3D
template <typename T>
inline void numpy_to_device_array_3d(const nb::ndarray<T, nb::c_contig> &array, core::cuda::DeviceArray3D<T> &device_array_3d) {
    if (array.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D NumPy array");
    }
    device_array_3d.upload(array.data(), array.shape(1), array.shape(0), array.shape(2)); // Upload raw pointer data into device array
}

#pragma endregion

} // namespace python_helper