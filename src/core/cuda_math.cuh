/*

cuda math functions

*/
#pragma once
#include <cuda_runtime.h>

namespace cuda_math {

#pragma region CONSTANTS

constexpr float SQRT2 = 1.4142135623730950488f;      // root of 2
constexpr float INV_SQRT2 = 0.70710678118654752440f; // inverse root of 2
constexpr float PI = 3.14159265358979323846f;
constexpr float GOLDEN_RATIO = 1.6180339887498948482f;

#pragma endregion

#pragma region POSMOD

// positive modulo for wrapping map coordinates
__device__ __forceinline__ int posmod(int i, int mod) {
    int result = i % mod;
    return result < 0 ? result + mod : result;
}
// float version, haven't checked yet
__device__ __forceinline__ float posmod(float x, float mod) {
    float result = fmodf(x, mod); // remainder in (-mod, mod)
    return result < 0.0f ? result + mod : result;
}

// positive modulo on int2
__device__ __forceinline__ int2 posmod(int2 pos, int2 mod) {
    return make_int2(posmod(pos.x, mod.x), posmod(pos.y, mod.y));
}

#pragma endregion

#pragma region CLAMP

// general clamp
template <typename T>
__device__ __forceinline__ T clamp(T value, T minimum, T maximum) {
    return value < minimum ? minimum : (value > maximum ? maximum : value);
}

// clamp integer to an index, ie range 8 => [0, 7]
__device__ inline int clamp_index(int i, int range) {
    return clamp(i, 0, range - 1);
}

__device__ inline int2 clamp_index(int2 pos, int2 range) {
    return make_int2(clamp_index(pos.x, range.x), clamp_index(pos.y, range.y));
}

#pragma endregion

#pragma region WRAP_OR_CLAMP

// // wrap or clamp an index, used for accesing map in range
// template <typename T>
// __device__ __forceinline__ T wrap_or_clamp_index(T i, T range, bool wrap) {
//     return wrap ? posmod(i, range) : clamp_index(i, range);
// }

// wrap or clamp for map access
__device__ __forceinline__ int wrap_or_clamp_index(int i, int range, bool wrap) {
    return wrap ? posmod(i, range) : clamp_index(i, range);
}

__device__ __forceinline__ int2 wrap_or_clamp_index(int2 pos, int2 range, bool wrap) {
    return wrap ? posmod(pos, range) : clamp_index(pos, range);
}

#pragma endregion

#pragma region VECTOR_OPS

// length of float2 vector
__host__ __device__ __forceinline__ float length(const float2 &vector) {
    return sqrt(vector.x * vector.x + vector.y * vector.y);
}

// dot product of two float2's
__host__ __device__ __forceinline__ float dot(const float2 &a, const float2 &b) {
    return a.x * b.x + a.y * b.y;
}

// 2D cross product (returns scalar z-component)
__host__ __device__ __forceinline__ float cross(const float2 &a, const float2 &b) {
    return a.x * b.y - a.y * b.x;
}

// normalize float2 vector to length of 1.0
__host__ __device__ __forceinline__ float2 normalize(const float2 &v) {
    float len = sqrtf(v.x * v.x + v.y * v.y);
    return (len > 1e-6f) ? make_float2(v.x / len, v.y / len) : make_float2(0.0f, 0.0f);
}
#pragma endregion



} // namespace cuda_math

#pragma region MATH_OPERATORS

// allow int2 == operator
__device__ __host__ inline bool operator==(const int2 &a, const int2 &b) {
    return (a.x == b.x) && (a.y == b.y);
}
// allow int2 != operator
__device__ __host__ inline bool operator!=(const int2 &a, const int2 &b) {
    return (a.x != b.x) || (a.y != b.y);
}

// allow int2 + operator
__device__ __host__ inline int2 operator+(const int2 &a, const int2 &b) {
    return make_int2(a.x + b.x, a.y + b.y);
}
// allow int2 - operator
__device__ __host__ inline int2 operator-(const int2 &a, const int2 &b) {
    return make_int2(a.x - b.x, a.y - b.y);
}

// allow dividing float2 by float
__host__ __device__ __forceinline__ float2 operator/(const float2 &a, float s) {
    return make_float2(a.x / s, a.y / s);
}

__host__ __device__ __forceinline__ float2 &operator/=(float2 &a, float s) {
    a.x /= s;
    a.y /= s;
    return a;
}

// allow multiplying float2 by float
__host__ __device__ __forceinline__ float2 operator*(const float2 &a, float s) {
    return make_float2(a.x * s, a.y * s);
}

__host__ __device__ __forceinline__ float2 operator*(float s, const float2 &a) {
    return make_float2(s * a.x, s * a.y);
}

__host__ __device__ __forceinline__ float2 &operator*=(float2 &a, float s) {
    a.x *= s;
    a.y *= s;
    return a;
}

#pragma endregion
