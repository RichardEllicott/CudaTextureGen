/*

conviniant loading from arrays of int's or floats

uses reinterpret_cast if possible

*/
#pragma once

#include <cuda_runtime.h>

#include "core/defines.h"

// #define INDEX_TYPE size_t
#define INDEX_TYPE int // note size_t might be technically correct but in practise we always have ints

namespace core::cuda::array {

#pragma region PACKED_VECTOR_ARRAY_ACCESSORS

#pragma region TRAITS

template <typename T>
struct vec_traits;

template <>
struct vec_traits<float2> {
    static constexpr int count = 2;
    static constexpr int alignment = 8; // 8‑byte aligned
    static constexpr bool can_reinterpret = true;
};

template <>
struct vec_traits<float3> {
    static constexpr int count = 3;
    static constexpr int alignment = 4; // only 4‑byte aligned
    static constexpr bool can_reinterpret = false;
};

template <>
struct vec_traits<float4> {
    static constexpr int count = 4;
    static constexpr int alignment = 16; // 16‑byte aligned
    static constexpr bool can_reinterpret = true;
};

template <>
struct vec_traits<int2> {
    static constexpr int count = 2;
    static constexpr int alignment = 8;
    static constexpr bool can_reinterpret = true;
};

template <>
struct vec_traits<int3> {
    static constexpr int count = 3;
    static constexpr int alignment = 4;
    static constexpr bool can_reinterpret = false;
};

template <>
struct vec_traits<int4> {
    static constexpr int count = 4;
    static constexpr int alignment = 16;
    static constexpr bool can_reinterpret = true;
};
#pragma endregion

#pragma region TEMPLATE

// Generic load for float2/3/4 and int2/3/4.
// Scalar = float, int, unsigned int, etc.
template <typename T, typename Scalar>
DH_INLINE T load(const Scalar *base, INDEX_TYPE idx) {
    if constexpr (vec_traits<T>::can_reinterpret) {
        return *reinterpret_cast<const T *>(&base[idx]); // fast path
    } else {
        const Scalar *p = &base[idx];
        if constexpr (std::is_same_v<T, float3>)
            return make_float3(p[0], p[1], p[2]);
        else
            return make_int3(p[0], p[1], p[2]);
    }
}

// Generic store for float2/3/4 and int2/3/4.
template <typename T, typename Scalar>
DH_INLINE void store(Scalar *base, INDEX_TYPE idx, T v) {
    if constexpr (vec_traits<T>::can_reinterpret) {
        *reinterpret_cast<T *>(&base[idx]) = v; // fast path
    } else {
        Scalar *p = &base[idx];
        p[0] = v.x;
        p[1] = v.y;
        p[2] = v.z;
    }
}

#pragma endregion

#pragma region SHORTCUTS

// load float2 from float array
DH_INLINE float2 load_float2(const float *base, INDEX_TYPE idx2) { return load<float2, float>(base, idx2); }

// load float3 from float array
DH_INLINE float3 load_float3(const float *base, INDEX_TYPE idx3) { return load<float3, float>(base, idx3); }

// load float4 from float array
DH_INLINE float4 load_float4(const float *base, INDEX_TYPE idx4) { return load<float4, float>(base, idx4); }

// load int2 from int array
DH_INLINE int2 load_int2(const int *base, INDEX_TYPE idx2) { return load<int2, int>(base, idx2); }

// load int3 from int array
DH_INLINE int3 load_int3(const int *base, INDEX_TYPE idx3) { return load<int3, int>(base, idx3); }

// load int4 from int array
DH_INLINE int4 load_int4(const int *base, INDEX_TYPE idx4) { return load<int4, int>(base, idx4); }

// --------------------------------------------------------------------------------------------------------------------------------

// store float2 to float array
DH_INLINE void store_float2(float *base, INDEX_TYPE idx2, float2 vec) { store<float2, float>(base, idx2, vec); }

// store float3 to float array
DH_INLINE void store_float3(float *base, INDEX_TYPE idx3, float3 vec) { store<float3, float>(base, idx3, vec); }

// store float4 to float array
DH_INLINE void store_float4(float *base, INDEX_TYPE idx4, float4 vec) { store<float4, float>(base, idx4, vec); }

// store int2 to int array
DH_INLINE void store_int2(int *base, INDEX_TYPE idx2, int2 vec) { store<int2, int>(base, idx2, vec); }

// store int3 to int array
DH_INLINE void store_int3(int *base, INDEX_TYPE idx3, int3 vec) { store<int3, int>(base, idx3, vec); }

// store int4 to int array
DH_INLINE void store_int4(int *base, INDEX_TYPE idx4, int4 vec) { store<int4, int>(base, idx4, vec); }

// ================================================================================================================================

#pragma endregion

#pragma endregion

} // namespace core::cuda::array

#undef INDEX_TYPE