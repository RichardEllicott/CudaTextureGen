/*

custom Cuda objects, designed to automaticly allocate and free memory, download etc

*/

#pragma once

#include "core/cuda/device_array.cuh"
#include "core/ref.h"
#include <array>

#include <cstddef> // size_t

#define D_INLINE __device__ __forceinline__           // device only functions
#define DH_INLINE __device__ __host__ __forceinline__ // device and host functions

namespace core::cuda::types {

// ================================================================================================================================

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

// --------------------------------------------------------------------------------------------------------------------------------

// ⚠️ not CUDA compatible
// templates allow usage in macros (which hate commas)
// usage:
//     FloatArray<8>

// template <std::size_t N>
// using FloatArray = std::array<float, N>;

// template <std::size_t N>
// using IntArray = std::array<int, N>;

// template <std::size_t N>
// using BoolArray = std::array<bool, N>;

// ================================================================================================================================
// [Custom CUDA compatible array]
// --------------------------------------------------------------------------------------------------------------------------------

template <typename T, size_t N>
struct Array {
    T data[N];

    // element access
    DH_INLINE T &operator[](size_t i) {
        return data[i];
    }

    DH_INLINE const T &operator[](std::size_t i) const {
        return data[i];
    }

    // equality
    DH_INLINE bool operator==(const Array &other) const {
        for (size_t i = 0; i < N; ++i)
            if (data[i] != other.data[i])
                return false;
        return true;
    }

    // inequality
    DH_INLINE bool operator!=(const Array &other) const {
        return !(*this == other);
    }

    // front/back (optional, but std::array-like)
    DH_INLINE T &front() { return data[0]; }
    DH_INLINE const T &front() const { return data[0]; }

    DH_INLINE T &back() { return data[N - 1]; }
    DH_INLINE const T &back() const { return data[N - 1]; }

    // data pointer
    DH_INLINE T *data_ptr() { return data; }
    DH_INLINE const T *data_ptr() const { return data; }

    // size (constexpr, host-only is fine here)
    static constexpr std::size_t size() { return N; }
};

template <std::size_t N>
using FloatArray = Array<float, N>;

template <std::size_t N>
using IntArray = Array<int, N>;

// custom bool version array, will be seen as bools either side
// (internally is byte so wastes some memory)

template <size_t N>
struct BoolArray {
    unsigned char data[N];

    DH_INLINE bool operator[](size_t i) const { return data[i] != 0; }
    DH_INLINE void set(size_t i, bool v) { data[i] = v ? 1 : 0; }

    // match CudaArray API
    DH_INLINE bool front() const { return data[0] != 0; }
    DH_INLINE bool back() const { return data[N - 1] != 0; }

    DH_INLINE unsigned char *data_ptr() { return data; }
    DH_INLINE const unsigned char *data_ptr() const { return data; }

    static constexpr size_t size() { return N; }

    // equality
    DH_INLINE bool operator==(const BoolArray &other) const {
        for (size_t i = 0; i < N; ++i)
            if (data[i] != other.data[i])
                return false;
        return true;
    }

    DH_INLINE bool operator!=(const BoolArray &other) const {
        return !(*this == other);
    }
};

// --------------------------------------------------------------------------------------------------------------------------------

} // namespace core::cuda::types