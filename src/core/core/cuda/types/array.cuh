/*

Custom CUDA compatible array


alternative to 

custom array type designed to be the same as std::array but works in kernels (__device__ __host__)
has a corrosponding required type caster as nanobind doesn't recognise out of the box
note seperate aray below for bool (as bool arrays can be unreliable on device)
use with xmacros as IntArray<8>, FloatArray<10>, BoolArray<3> etc...



*/
#pragma once

#include <cstddef> // size_t

#include "core/defines.h"

namespace core::cuda::types {

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

template <std::size_t N>
using CharArray = Array<char, N>;

// ================================================================================================================================
// [Custom bool array that converts to char/int on the c++ side, shows as bool on python side]
// --------------------------------------------------------------------------------------------------------------------------------

#define BOOL_ARRAY_CODE_ROUTE 1 // 0 char array, 1 int array
#if BOOL_ARRAY_CODE_ROUTE == 0

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

#elif BOOL_ARRAY_CODE_ROUTE == 1

template <size_t N>
struct BoolArray {
    int data[N];

    DH_INLINE bool operator[](size_t i) const { return data[i] != 0; }
    DH_INLINE void set(size_t i, bool v) { data[i] = v ? 1 : 0; }

    // match CudaArray API
    DH_INLINE bool front() const { return data[0] != 0; }
    DH_INLINE bool back() const { return data[N - 1] != 0; }

    DH_INLINE int *data_ptr() { return data; }
    DH_INLINE const int *data_ptr() const { return data; }

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

#endif
#undef BOOL_ARRAY_CODE_ROUTE

// ================================================================================================================================

} // namespace core::cuda::types