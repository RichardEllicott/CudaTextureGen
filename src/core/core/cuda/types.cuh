/*

custom Cuda objects, designed to automaticly allocate and free memory, download etc

*/

#pragma once

#include <array>
#include <cstddef> // size_t

#include "core/cuda/device_array.cuh"
#include "core/defines.h"
#include "core/ref.h"
#include "types/array.cuh" // moved my cuda array here

namespace core::cuda::types {

// // (LABEL, TYPE, DIMENSIONS)
// #define XMACRO_ALPHA_1              \
//     X(DeviceArrayFloat1D, float, 1) \
//     X(DeviceArrayFloat2D, float, 2) \
//     X(DeviceArrayFloat3D, float, 3)

// #define X(LABEL, TYPE, DIMENSIONS) \
//     using LABEL = core::cuda::DeviceArray<TYPE, DIMENSIONS>;
// XMACRO_ALPHA_1
// #undef X

// ================================================================================================================================
// [Device Array Refs]
// --------------------------------------------------------------------------------------------------------------------------------

// aliases ⚠️ WANT TO PUT THESE WHERE DEVICE ARRAY LIVES BUT CODE RELIES ON THE OLD TEMPLATES!!!
template <typename T>
using DeviceArray1D = DeviceArray<T, 1>;

template <typename T>
using DeviceArray2D = DeviceArray<T, 2>;

template <typename T>
using DeviceArray3D = DeviceArray<T, 3>;

template <typename T>
using DeviceArray4D = DeviceArray<T, 4>;

// --------------------------------------------------------------------------------------------------------------------------------

using DeviceArrayFloat1D = DeviceArray<float, 1>;
using DeviceArrayFloat2D = DeviceArray<float, 2>;
using DeviceArrayFloat3D = DeviceArray<float, 3>;
// using DeviceArrayFloat4D = DeviceArray<float, 4>;

using DeviceArrayInt1D = DeviceArray<int, 1>;
using DeviceArrayInt2D = DeviceArray<int, 2>;
using DeviceArrayInt3D = DeviceArray<int, 3>;
// using DeviceArrayInt4D = DeviceArray<int, 4>;

using DeviceArrayChar1D = DeviceArray<char, 1>;
using DeviceArrayChar2D = DeviceArray<char, 2>;
using DeviceArrayChar3D = DeviceArray<char, 3>;
// using DeviceArrayChar4D = DeviceArray<char, 4>;

// --------------------------------------------------------------------------------------------------------------------------------

using DeviceArrayFloat1DRef = Ref<DeviceArrayFloat1D>;
using DeviceArrayFloat2DRef = Ref<DeviceArrayFloat3D>;
using DeviceArrayFloat3DRef = Ref<DeviceArrayFloat3D>;
// using DeviceArrayFloat4DRef = Ref<DeviceArrayFloat4D>;

using DeviceArrayInt1DRef = Ref<DeviceArrayInt1D>;
using DeviceArrayInt2DRef = Ref<DeviceArrayInt2D>;
using DeviceArrayInt3DRef = Ref<DeviceArrayInt3D>;
// using DeviceArrayInt4DRef = Ref<DeviceArrayInt4D>;

using DeviceArrayChar1DRef = Ref<DeviceArrayChar1D>;
using DeviceArrayChar2DRef = Ref<DeviceArrayChar2D>;
using DeviceArrayChar3DRef = Ref<DeviceArrayChar3D>;
// using DeviceArrayChar4DRef = Ref<DeviceArrayChar4D>;

} // namespace core::cuda::types