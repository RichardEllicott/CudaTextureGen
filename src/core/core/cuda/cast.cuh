/*


*/

#pragma once

#include <array>
#include <cuda_runtime.h>

#define D_INLINE __device__ __forceinline__           // device only functions
#define DH_INLINE __device__ __host__ __forceinline__ // device and host functions

namespace core::cuda::cast {

#pragma region CAST_MANUAL

// // ================================================================================================================================
// // [Casting]
// // --------------------------------------------------------------------------------------------------------------------------------
// // std::array<float, 2> => float2
// inline float2 to_float2(const std::array<float, 2> &val) {
//     return {val[0], val[1]};
// }
// // std::array<float, 3> => float3
// inline float3 to_float3(const std::array<float, 3> &val) {
//     return {val[0], val[1], val[2]};
// }
// // std::array<float, 4> => float4
// inline float4 to_float4(const std::array<float, 4> &val) {
//     return {val[0], val[1], val[2], val[3]};
// }
// // --------------------------------------------------------------------------------------------------------------------------------
// // std::array<int, 2> => int2
// inline int2 to_int2(const std::array<int, 2> &val) {
//     return {val[0], val[1]};
// }
// // std::array<int, 3> => int3
// inline int3 to_int3(const std::array<int, 3> &val) {
//     return {val[0], val[1], val[2]};
// }
// // std::array<int, 4> => int4
// inline int4 to_int4(const std::array<int, 4> &val) {
//     return {val[0], val[1], val[2], val[3]};
// }
// // --------------------------------------------------------------------------------------------------------------------------------
// // float2 => int2
// DH_INLINE int2 to_int2(const float2 &f) {
//     return {(int)f.x, (int)f.y};
// }
// // float3 => int3
// DH_INLINE int3 to_int3(const float3 &f) {
//     return {(int)f.x, (int)f.y, (int)f.z};
// }
// // float4 => int4
// DH_INLINE int4 to_int4(const float4 &f) {
//     return {(int)f.x, (int)f.y, (int)f.z, (int)f.w};
// }
// // --------------------------------------------------------------------------------------------------------------------------------
// // std::array<bool, 2> => int2
// inline int2 to_int2(const std::array<bool, 2> &a) {
//     return make_int2(
//         a[0] ? 1 : 0,
//         a[1] ? 1 : 0);
// }
// // std::array<bool, 3> => int3
// inline int3 to_int3(const std::array<bool, 3> &a) {
//     return make_int3(
//         a[0] ? 1 : 0,
//         a[1] ? 1 : 0,
//         a[2] ? 1 : 0);
// }
// // std::array<bool, 4> => int4
// inline int4 to_int4(const std::array<bool, 4> &a) {
//     return make_int4(
//         a[0] ? 1 : 0,
//         a[1] ? 1 : 0,
//         a[2] ? 1 : 0,
//         a[3] ? 1 : 0);
// }
// // --------------------------------------------------------------------------------------------------------------------------------

#pragma endregion

#pragma region TEMPLATE

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

DH_INLINE inline int2 to_int2(const float2 &v) {
    return make_int2(static_cast<int>(v.x),
                     static_cast<int>(v.y));
}

DH_INLINE inline int3 to_int3(const float3 &v) {
    return make_int3(static_cast<int>(v.x),
                     static_cast<int>(v.y),
                     static_cast<int>(v.z));
}

DH_INLINE inline int4 to_int4(const float4 &v) {
    return make_int4(static_cast<int>(v.x),
                     static_cast<int>(v.y),
                     static_cast<int>(v.z),
                     static_cast<int>(v.w));
}

#pragma endregion

} // namespace core::cuda::cast