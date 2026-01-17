/*

casting various arrays and types to cuda types like int3 etc

*/

#pragma once

#include <array>
#include <cuda_runtime.h>

// #include "core/cuda/types.cuh"
#include "types.cuh"

#define D_INLINE __device__ __forceinline__           // device only functions
#define DH_INLINE __device__ __host__ __forceinline__ // device and host functions

#define CASTING_CODE_ROUTE 0

namespace core::cuda::cast {

#if CASTING_CODE_ROUTE == 0

// ================================================================================================================================
// [Casting]
// --------------------------------------------------------------------------------------------------------------------------------
// [float]
// --------------------------------------------------------------------------------------------------------------------------------
// std::array<float, 2> => float2
inline float2 to_float2(const std::array<float, 2> &arr) { return make_float2(arr[0], arr[1]); }
// std::array<float, 3> => float3
inline float3 to_float3(const std::array<float, 3> &arr) { return make_float3(arr[0], arr[1], arr[2]); }
// std::array<float, 4> => float4
inline float4 to_float4(const std::array<float, 4> &arr) { return make_float4(arr[0], arr[1], arr[2], arr[3]); }
// --------------------------------------------------------------------------------------------------------------------------------
// int2 => float2
DH_INLINE float2 to_float2(int2 v) { return make_float2(v.x, v.y); }
// int3 => float3
DH_INLINE float3 to_float3(int3 v) { return make_float3(v.x, v.y, v.z); }
// int4 => float4
DH_INLINE float4 to_float4(int4 v) { return make_float4(v.x, v.y, v.z, v.w); }
// --------------------------------------------------------------------------------------------------------------------------------
// [int]
// --------------------------------------------------------------------------------------------------------------------------------
// std::array<int, 2> => int2
inline int2 to_int2(const std::array<size_t, 2> &arr) { return make_int2(arr[0], arr[1]); }
// std::array<int, 3> => int3
inline int3 to_int3(const std::array<size_t, 3> &arr) { return make_int3(arr[0], arr[1], arr[2]); }
// std::array<int, 4> => int4
inline int4 to_int4(const std::array<size_t, 4> &arr) { return make_int4(arr[0], arr[1], arr[2], arr[3]); }
// --------------------------------------------------------------------------------------------------------------------------------
// BoolArray<2> => int2
DH_INLINE int2 to_int2(const core::cuda::types::BoolArray<2> &arr) { return make_int2(arr[0], arr[1]); }
// BoolArray<3> => int3
DH_INLINE int3 to_int3(const core::cuda::types::BoolArray<3> &arr) { return make_int3(arr[0], arr[1], arr[2]); }
// BoolArray<4> => int4
DH_INLINE int4 to_int4(const core::cuda::types::BoolArray<4> &arr) { return make_int4(arr[0], arr[1], arr[2], arr[3]); }
// --------------------------------------------------------------------------------------------------------------------------------
// float2 => int2
DH_INLINE int2 to_int2(const float2 &v) { return make_int2(v.x, v.y); }
// float3 => int3
DH_INLINE int3 to_int3(const float3 &v) { return make_int3(v.x, v.y, v.z); }
// float4 => int4
DH_INLINE int4 to_int4(const float4 &v) { return make_int4(v.x, v.y, v.z, v.w); }
// --------------------------------------------------------------------------------------------------------------------------------
// std::array<bool, 2> => int2
inline int2 to_int2(const std::array<bool, 2> &a) { return make_int2(a[0], a[1]); }
// std::array<bool, 3> => int3
inline int3 to_int3(const std::array<bool, 3> &a) { return make_int3(a[0], a[1], a[2]); }
// std::array<bool, 4> => int4
inline int4 to_int4(const std::array<bool, 4> &a) { return make_int4(a[0], a[1], a[2], a[3]); }
// --------------------------------------------------------------------------------------------------------------------------------

#endif

#if CASTING_CODE_ROUTE == 1 // too... macro version?

#endif

#if CASTING_CODE_ROUTE == 2 // still flaky

// ================================================================================================================================
// [Simple Templates]
// --------------------------------------------------------------------------------------------------------------------------------

// cast to int2
template <typename T>
DH_INLINE int2 to_int2(const T &v) {
    return make_int2(
        (int)v.x,
        (int)v.y);
}

// cast to int3
template <typename T>
DH_INLINE int3 to_int3(const T &v) {
    return make_int3(
        (int)v.x,
        (int)v.y,
        (int)v.z);
}

// cast to int4
template <typename T>
DH_INLINE int4 to_int4(const T &v) {
    return make_int4(
        (int)v.x,
        (int)v.y,
        (int)v.z,
        (int)v.w);
}

// --------------------------------------------------------------------------------------------------------------------------------

// std::array<T,2> → int2
template <typename T>
DH_INLINE int2 to_int2(const std::array<T, 2> &arr) {
    return make_int2(
        (int)arr[0],
        (int)arr[1]);
}

// std::array<T,3> → int3
template <typename T>
DH_INLINE int3 to_int3(const std::array<T, 3> &arr) {
    return make_int3(
        (int)arr[0],
        (int)arr[1],
        (int)arr[2]);
}

// std::array<T,4> → int4
template <typename T>
DH_INLINE int4 to_int4(const std::array<T, 4> &arr) {
    return make_int4(
        (int)arr[0],
        (int)arr[1],
        (int)arr[2],
        (int)arr[3]);
}

// ================================================================================================================================

// cast to float2
template <typename T>
DH_INLINE float2 to_float2(const T &v) {
    return make_float2(
        (float)v.x,
        (float)v.y);
}

// cast to float3
template <typename T>
DH_INLINE float3 to_float3(const T &v) {
    return make_float3(
        (float)v.x,
        (float)v.y,
        (float)v.z);
}

// cast to float4
template <typename T>
DH_INLINE float4 to_float4(const T &v) {
    return make_float4(
        (float)v.x,
        (float)v.y,
        (float)v.z,
        (float)v.w);
}

// --------------------------------------------------------------------------------------------------------------------------------

// std::array<T,2> → float2
template <typename T>
DH_INLINE float2 to_float2(const std::array<T, 2> &arr) {
    return make_float2(
        (float)arr[0],
        (float)arr[1]);
}

// std::array<T,3> → float3
template <typename T>
DH_INLINE float3 to_float3(const std::array<T, 3> &arr) {
    return make_float3(
        (float)arr[0],
        (float)arr[1],
        (float)arr[2]);
}

// std::array<T,4> → float4
template <typename T>
DH_INLINE float4 to_float4(const std::array<T, 4> &arr) {
    return make_float4(
        (float)arr[0],
        (float)arr[1],
        (float)arr[2],
        (float)arr[3]);
}

// --------------------------------------------------------------------------------------------------------------------------------

#endif

#if CASTING_CODE_ROUTE == 3 // this template turned out too compliated for GCC

// -----------------------------------------------------------------------------
// Helpers: build CUDA intN vectors
// -----------------------------------------------------------------------------

template <int N>
struct make_intN;

template <>
struct make_intN<2> {
    static inline int2 make(int x, int y) {
        return make_int2(x, y);
    }
};

template <>
struct make_intN<3> {
    static inline int3 make(int x, int y, int z) {
        return make_int3(x, y, z);
    }
};

template <>
struct make_intN<4> {
    static inline int4 make(int x, int y, int z, int w) {
        return make_int4(x, y, z, w);
    }
};

// -----------------------------------------------------------------------------
// std::array<T, N> -> intN  (host-side only; safe for bool)
// -----------------------------------------------------------------------------

template <typename T, int N>
inline auto to_intN(const std::array<T, N> &a) {
    if constexpr (N == 2) {
        return make_intN<2>::make(
            static_cast<int>(a[0]),
            static_cast<int>(a[1]));
    }
    if constexpr (N == 3) {
        return make_intN<3>::make(
            static_cast<int>(a[0]),
            static_cast<int>(a[1]),
            static_cast<int>(a[2]));
    }
    if constexpr (N == 4) {
        return make_intN<4>::make(
            static_cast<int>(a[0]),
            static_cast<int>(a[1]),
            static_cast<int>(a[2]),
            static_cast<int>(a[3]));
    }
}

// Convenience aliases if you like explicit names:

inline int2 to_int2(const std::array<bool, 2> &a) {
    return to_intN<bool, 2>(a);
}

inline int3 to_int3(const std::array<bool, 3> &a) {
    return to_intN<bool, 3>(a);
}

inline int4 to_int4(const std::array<bool, 4> &a) {
    return to_intN<bool, 4>(a);
}

// You can add int/float variants similarly if desired:
// inline int3 to_int3(const std::array<int, 3>& a) { return to_intN<int, 3>(a); }

// -----------------------------------------------------------------------------
// floatN -> intN  (host or device; POD CUDA vector types)
// -----------------------------------------------------------------------------

// DH_INLINE ??? can't do this?

inline int2 to_int2(const float2 &v) {
    return make_int2(static_cast<int>(v.x),
                     static_cast<int>(v.y));
}

inline int3 to_int3(const float3 &v) {
    return make_int3(static_cast<int>(v.x),
                     static_cast<int>(v.y),
                     static_cast<int>(v.z));
}

inline int4 to_int4(const float4 &v) {
    return make_int4(static_cast<int>(v.x),
                     static_cast<int>(v.y),
                     static_cast<int>(v.z),
                     static_cast<int>(v.w));
}

#endif

} // namespace core::cuda::cast

#undef CASTING_CODE_ROUTE