/*

convert numpy arrays to and from DeviceArray's

new DeviceArrayN pattern

*/
#pragma once

#include "cuda_types.cuh"
#include "numpy.h" // numpy helper in same folder
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <cstring> // required for std::memcpy in linux (not windows)

namespace nanobind::helper::numpy {

namespace nb = nanobind; // shortcut

#pragma region DEVICE_ARRAY_N

// convert DeviceArrayN to ndarray<T>
template <typename T, int Dim>
inline nb::ndarray<nb::numpy, T> to_array(const core::cuda::DeviceArrayN<T, Dim> &device_array) {
    // Internal order: width, height, depth...
    auto shape = device_array.dimensions();

    // Flip for NumPy if Dim >= 2
    if constexpr (Dim >= 2) {
        std::swap(shape[0], shape[1]);
    }

    auto array = get_array<T, Dim>(shape);
    device_array.download(array.data());
    return array;
}

// copy numpy array to DeviceArrayN
template <typename T, int Dim>
inline void to_device_array(const nb::ndarray<T, nb::c_contig> &source, core::cuda::DeviceArrayN<T, Dim> &destination) {
    if (source.ndim() != Dim)
        throw std::runtime_error("Input must be a " + std::to_string(Dim) + "D NumPy array");

    // Gather NumPy shape (height, width, depth...)
    std::array<size_t, Dim> np_dims{};
    for (int i = 0; i < Dim; ++i)
        np_dims[i] = static_cast<size_t>(source.shape(i));

    // Flip back to internal order (width, height, depth)
    std::array<size_t, Dim> internal_dims = np_dims;
    if constexpr (Dim >= 2)
        std::swap(internal_dims[0], internal_dims[1]);

    destination.resize(internal_dims);
    destination.upload(source.data(), internal_dims);
}

#pragma endregion

#pragma region DEVICE_ARRAY_N2D

#pragma endregion

#pragma region DEVICE_ARRAY_N3D

#pragma endregion

#pragma region PYTHON_NONE_SUPPORT

// ⚠️ decided not to use these and leave the program more strongly typed, to zero an array, send in a zero sized numpy array
// convert or free the device if None... a more liberal pattern accepting nb::object
template <typename T, int Dim>
inline void python_to_device_array(nb::object obj, core::cuda::DeviceArrayN<T, Dim> &device_array) {
    if (obj.is_none()) {
        device_array.free_device(); // Free the device array if Python passed None
    } else {
        // Cast Python object to a NumPy ndarray of type T
        auto array = nb::cast<nb::ndarray<T, nb::c_contig>>(obj);
        to_device_array(array, device_array);
    }
}

// Returns None if device array empty
template <typename T, int Dim>
inline nb::object device_array_to_python(const core::cuda::DeviceArrayN<T, Dim> &device_array) {
    if (device_array.empty()) {
        return nb::none(); // If the device array has no data, return Python None
    } else {
        auto array = to_array(device_array); // download device array to numpy array
        return nb::cast(array);              // cast required
    }
}

#pragma endregion

} // namespace nanobind::helper::numpy
