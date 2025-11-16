/*

convert numpy arrays to and from DeviceArray's


device_array_to_numpy(device_array) // returns a numpy array

numpy_to_device_array()

*/
#pragma once

#include "cuda_types.cuh"
#include "numpy.h" // numpy helper in same folder
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <cstring> // required for std::memcpy in linux (not windows)

namespace python_helper {

namespace nb = nanobind; // shortcut

#pragma region DEVICE_ARRAY_1D

// download a DeviceArray's data to a new numpy array
template <typename T>
inline nb::ndarray<nb::numpy, T> device_array_to_numpy(const core::cuda::DeviceArray<T> &device_array) {
    auto array = python_helper::get_numpy_array<T>(device_array.size()); // create numpy array
    device_array.download(array.data());                                 // download the data into the numpy array
    return array;
}

// upload a numpy array's data to a DeviceArray
template <typename T>
inline void numpy_to_device_array(const nb::ndarray<T, nb::c_contig> &source, core::cuda::DeviceArray<T> &destination) {
    if (source.ndim() != 1) {
        throw std::runtime_error("Input must be a 1D NumPy array");
    }
    destination.upload(source.data(), source.shape(0)); // Upload raw pointer data into device array
}

#pragma endregion

#pragma region DEVICE_ARRAY_2D

// download a DeviceArray2D's data to a new numpy array
template <typename T>
inline nb::ndarray<nb::numpy, T> device_array_to_numpy(const core::cuda::DeviceArray2D<T> &device_array) {
    auto array = python_helper::get_numpy_array<T>(device_array.height(), device_array.width()); // create numpy array
    device_array.download(array.data());                                                            // download the data into the numpy array
    return array;
}

// upload a numpy array's data to a DeviceArray2D
template <typename T>
inline void numpy_to_device_array(const nb::ndarray<T, nb::c_contig> &source, core::cuda::DeviceArray2D<T> &destination) {
    if (source.ndim() != 2) {
        throw std::runtime_error("Input must be a 2D NumPy array");
    }
    destination.upload(source.data(), source.shape(1), source.shape(0)); // Upload raw pointer data into device array
}

#pragma endregion

#pragma region DEVICE_ARRAY_3D

// download a DeviceArray3D's data to a new numpy array
template <typename T>
inline nb::ndarray<nb::numpy, T> device_array_to_numpy(const core::cuda::DeviceArray3D<T> &device_array) {
    auto array = python_helper::get_numpy_array<T>(device_array.height(), device_array.width(), device_array.depth()); // create numpy array
    device_array.download(array.data());                                                                                     // download the data into the numpy array
    return array;
}

// upload a numpy array's data to a DeviceArray3D
template <typename T>
inline void numpy_to_device_array(const nb::ndarray<T, nb::c_contig> &source, core::cuda::DeviceArray3D<T> &destination) {
    if (source.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D NumPy array");
    }
    destination.upload(source.data(), source.shape(1), source.shape(0), source.shape(2)); // Upload raw pointer data into device array
}

#pragma endregion

} // namespace python_helper