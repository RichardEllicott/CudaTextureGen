/*

custom Cuda objects, designed to automaticly allocate and free memory, download etc

*/

#pragma once

#include "core/cuda/device_array.cuh"
#include "core/ref.h"
#include <array>
// #include "cast.cuh" // converting types


#define D_INLINE __device__ __forceinline__           // device only functions
#define DH_INLINE __device__ __host__ __forceinline__ // device and host functions

namespace core::cuda::types {

// standard DeviceArray Refs
using RefDeviceArrayFloat1D = core::Ref<core::cuda::DeviceArray<float, 1>>;
using RefDeviceArrayFloat2D = core::Ref<core::cuda::DeviceArray<float, 2>>;
using RefDeviceArrayFloat3D = core::Ref<core::cuda::DeviceArray<float, 3>>;

using RefDeviceArrayInt1D = core::Ref<core::cuda::DeviceArray<int, 1>>;
using RefDeviceArrayInt2D = core::Ref<core::cuda::DeviceArray<int, 2>>;
using RefDeviceArrayInt3D = core::Ref<core::cuda::DeviceArray<int, 3>>;

// // std::array aliases
// // using Float2 = std::array<float, 2>;
// // using Float3 = std::array<float, 3>;
// // using Float4 = std::array<float, 4>;
// // using Float5 = std::array<float, 5>;
// // using Float6 = std::array<float, 6>;
// // using Float7 = std::array<float, 7>;
// // using Float8 = std::array<float, 8>;

// using Int2 = std::array<int, 2>;
// using Int3 = std::array<int, 3>;
// using Int4 = std::array<int, 4>;
// using Int5 = std::array<int, 5>;
// using Int6 = std::array<int, 6>;
// using Int7 = std::array<int, 7>;
// using Int8 = std::array<int, 8>;

// //

// xmacro works but intelisense chokes, maybe use Template

// // (DIMENSIONS)
// #define ARRAY_TYPE_NUMBERS \
//     X(2)             \
//     X(3)             \
//     X(4)             \
//     X(5)             \
//     X(6)             \
//     X(7)             \
//     X(8)

// #define X(DIMENSIONS) \
//     using Float##DIMENSIONS = std::array<float, DIMENSIONS>;
// ARRAY_TYPE_NUMBERS
// #undef X

// #undef TYPE_NUMBERS

//
//

// templates allow usage in macros (which hate commas)
// usage:
//     FloatArray<8>
template <std::size_t N>
using FloatArray = std::array<float, N>;

template <std::size_t N>
using IntArray = std::array<int, N>;

template <std::size_t N>
using BoolArray = std::array<bool, N>;



// --------------------------------------------------------------------------------------------------------------------------------

} // namespace core::cuda::types