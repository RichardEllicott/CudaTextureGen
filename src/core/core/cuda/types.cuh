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

// ================================================================================================================================
// [Device Array Refs]
// --------------------------------------------------------------------------------------------------------------------------------

using DeviceArrayFloat1DRef = core::Ref<core::cuda::DeviceArray<float, 1>>;
using DeviceArrayFloat2DRef = core::Ref<core::cuda::DeviceArray<float, 2>>;
using DeviceArrayFloat3DRef = core::Ref<core::cuda::DeviceArray<float, 3>>;

using DeviceArrayInt1DRef = core::Ref<core::cuda::DeviceArray<int, 1>>;
using DeviceArrayInt2DRef = core::Ref<core::cuda::DeviceArray<int, 2>>;
using DeviceArrayInt3DRef = core::Ref<core::cuda::DeviceArray<int, 3>>;

// doubled (refactoring away)
using RefDeviceArrayFloat1D = DeviceArrayFloat1DRef;
using RefDeviceArrayFloat2D = DeviceArrayFloat2DRef;
using RefDeviceArrayFloat3D = DeviceArrayFloat3DRef;

using RefDeviceArrayInt1D = DeviceArrayInt1DRef;
using RefDeviceArrayInt2D = DeviceArrayInt2DRef;
using RefDeviceArrayInt3D = DeviceArrayInt3DRef;

} // namespace core::cuda::types