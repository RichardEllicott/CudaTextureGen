/*

custom Cuda objects, designed to automaticly allocate and free memory, download etc

*/

#pragma once

#include "core/cuda/device_array.cuh"
#include "core/ref.h"
#include <array>

namespace core::cuda::types {

// standard DeviceArray Refs
using RefDeviceArrayFloat1D = core::Ref<core::cuda::DeviceArray<float, 1>>;
using RefDeviceArrayFloat2D = core::Ref<core::cuda::DeviceArray<float, 2>>;
using RefDeviceArrayFloat3D = core::Ref<core::cuda::DeviceArray<float, 3>>;

using RefDeviceArrayInt1D = core::Ref<core::cuda::DeviceArray<int, 1>>;
using RefDeviceArrayInt2D = core::Ref<core::cuda::DeviceArray<int, 2>>;
using RefDeviceArrayInt3D = core::Ref<core::cuda::DeviceArray<int, 3>>;

// std::array aliases
using Float2 = std::array<float, 2>;
using Float3 = std::array<float, 3>;
using Float4 = std::array<float, 4>;
using Float5 = std::array<float, 5>;
using Float6 = std::array<float, 6>;
using Float7 = std::array<float, 7>;
using Float8 = std::array<float, 8>;

using Int2 = std::array<int, 2>;
using Int3 = std::array<int, 3>;
using Int4 = std::array<int, 4>;
using Int5 = std::array<int, 5>;
using Int6 = std::array<int, 6>;
using Int7 = std::array<int, 7>;
using Int8 = std::array<int, 8>;

template <std::size_t N>
using FloatArray = std::array<float, N>;

template <std::size_t N>
using IntArray = std::array<int, N>;

// host side helpers to convert to cuda versions

// Float2 => float2
inline float2 to_float2(const Float2 &val) { return {val[0], val[1]}; }
// Float3 => float3
inline float3 to_float3(const Float3 &val) { return {val[0], val[1], val[2]}; }
// Float4 => float4
inline float4 to_float4(const Float4 &val) { return {val[0], val[1], val[2], val[3]}; }

// Int2 => int2
inline int2 to_int2(const Int2 &val) { return {val[0], val[1]}; }
// Int3 => int3
inline int3 to_int3(const Int3 &val) { return {val[0], val[1], val[2]}; }
// Int4 => int4
inline int4 to_int4(const Int4 &val) { return {val[0], val[1], val[2], val[3]}; }

} // namespace core::cuda::types