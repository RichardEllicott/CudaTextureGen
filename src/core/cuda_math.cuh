/*

cuda math functions

*/
#pragma once
#include <cuda_runtime.h>

#define D_INLINE __device__ __forceinline__
#define DH_INLINE __device__ __host__ __forceinline__

namespace cuda_math {

#pragma region CONSTANTS

// device code can't see constexpr
constexpr float SQRT2 = 1.4142135623730950488f;      // root of 2
constexpr float INV_SQRT2 = 0.70710678118654752440f; // inverse root of 2
constexpr float PI = 3.14159265358979323846f;
constexpr float GOLDEN_RATIO = 1.6180339887498948482f;

#pragma endregion

#pragma region IDX

// pos to idx, note if we have layers (usually colour channels) we multiply the idx, then (R,G,B) = (idx+0, idx+1, idx+2)
DH_INLINE int pos_to_idx(int2 pos, int map_width, int layers = 1) {
    return (pos.y * map_width + pos.x) * layers;
}

// pos to idx formula
DH_INLINE int pos_to_idx(int x, int y, int map_width, int layers = 1) {
    return (y * map_width + x) * layers;
}

#pragma endregion

#pragma region POSMOD

// positive modulo for wrapping map coordinates
DH_INLINE int posmod(int i, int mod) {
    int result = i % mod;
    return result < 0 ? result + mod : result;
}

// float version, haven't checked yet
DH_INLINE float posmod(float x, float mod) {
    float result = fmodf(x, mod); // remainder in (-mod, mod)
    return result < 0.0f ? result + mod : result;
}

// positive modulo on int2
DH_INLINE int2 posmod(int2 pos, int2 mod) {
    return make_int2(posmod(pos.x, mod.x), posmod(pos.y, mod.y));
}

#pragma endregion

#pragma region CLAMP

// general clamp
template <typename T>
DH_INLINE T clamp(T value, T minimum, T maximum) {
    return value < minimum ? minimum : (value > maximum ? maximum : value);
}

// clamp integer to an index, ie range 8 => [0, 7]
DH_INLINE int clamp_index(int i, int range) {
    return clamp(i, 0, range - 1);
}

// clamp int2 to an index, ie range 8 => [0, 7]
DH_INLINE int2 clamp_index(int2 pos, int2 range) {
    return make_int2(clamp_index(pos.x, range.x), clamp_index(pos.y, range.y));
}

#pragma endregion

#pragma region WRAP_OR_CLAMP

// // wrap or clamp an index, used for accesing map in range
// template <typename T>
// __device__ __forceinline__ T wrap_or_clamp_index(T i, T range, bool wrap) {
//     return wrap ? posmod(i, range) : clamp_index(i, range);
// }

// wrap or clamp int for map access
DH_INLINE int wrap_or_clamp_index(int i, int range, bool wrap) {
    return wrap ? posmod(i, range) : clamp_index(i, range);
}

// wrap or clamp int2 for map access
DH_INLINE int2 wrap_or_clamp_index(int2 pos, int2 range, bool wrap) {
    return wrap ? posmod(pos, range) : clamp_index(pos, range);
}

#pragma endregion

#pragma region VECTOR_OPS

// length of float2 vector
DH_INLINE float length(const float2 &vector) {
    return sqrt(vector.x * vector.x + vector.y * vector.y);
}

// dot product of two float2's
DH_INLINE float dot(const float2 &a, const float2 &b) {
    return a.x * b.x + a.y * b.y;
}

// 2D cross product (returns scalar z-component)
DH_INLINE float cross(const float2 &a, const float2 &b) {
    return a.x * b.y - a.y * b.x;
}

// normalize float2 vector to length of 1.0
DH_INLINE float2 normalize(const float2 &v) {
    float len = sqrtf(v.x * v.x + v.y * v.y);
    return (len > 1e-6f) ? make_float2(v.x / len, v.y / len) : make_float2(0.0f, 0.0f);
}

#pragma endregion

#pragma region HASH

// integer hash (based on MurmurHash3 finalizer)
DH_INLINE int hash_int(int x, int y, int z, int seed) {
    int n = x + y * 374761393 + z * 668265263 + seed * 1274126177;

    n ^= n >> 16;
    n *= 0x85ebca6b;
    n ^= n >> 13;
    n *= 0xc2b2ae35;
    n ^= n >> 16;

    return n; // can be negative
}

// integer hash (based on MurmurHash3 finalizer)
DH_INLINE uint32_t hash_uint(uint32_t x, uint32_t y, uint32_t z, uint32_t seed) {
    uint32_t n = x + y * 374761393u + z * 668265263u + seed * 1274126177u;

    n ^= n >> 16;
    n *= 0x85ebca6bu;
    n ^= n >> 13;
    n *= 0xc2b2ae35u;
    n ^= n >> 16;

    return n; // full 32-bit unsigned result
}

// float from [0,1]
DH_INLINE float hash_float(uint32_t hash) {
    return static_cast<float>(hash) / pow(2.0f, 32.0f); // Scale to [0,1]
}

// float from [0,1]
DH_INLINE float hash_float(uint32_t x, uint32_t y, uint32_t z, uint32_t seed) {
    return hash_float(hash_uint(x, y, z, seed));
}

// float from [-1,1]
DH_INLINE float hash_float_signed(int hash) {
    return static_cast<float>(hash) / pow(2.0f, 31.0f); // Scale to [-1,1).
}

// float from [-1,1] range:
DH_INLINE float hash_float_signed(int x, int y, int z, int seed) {
    return hash_float_signed(hash_int(x, y, z, seed));
}

// take in a hash, extract a bool (set index from 0 to 31)
DH_INLINE bool hash_to_bool(int hash, int index = 0) {
    return (hash >> index) & 1u;
}

// get 4 random float's from one 32 bit hash, they are not so random though with about 255 possible values
// set byte_index from 0-3
DH_INLINE float hash_to_4randf(uint32_t h, int byte_index) {
    uint32_t byte = (h >> (8 * byte_index)) & 0xFFu;
    return (float(byte) / 127.5f) - 1.0f; // // map [0,255] to [-1,1]
}

#pragma endregion

#pragma region SMOOTHING_AND_SATURATION

// soft ceiling to avoid clipping artifacts where sharpness controls how sharp or gentle the saturation curve is.
DH_INLINE float soft_saturate(float value, float ceiling, float sharpness = 1.0) {
    return ceiling * tanh((value / ceiling) * sharpness);
}

#pragma endregion

} // namespace cuda_math

#pragma region MATH_OPERATORS

// allow int2 == operator
DH_INLINE bool operator==(const int2 &a, const int2 &b) {
    return (a.x == b.x) && (a.y == b.y);
}
// allow int2 != operator
DH_INLINE bool operator!=(const int2 &a, const int2 &b) {
    return (a.x != b.x) || (a.y != b.y);
}

// allow int2 + operator
DH_INLINE int2 operator+(const int2 &a, const int2 &b) {
    return make_int2(a.x + b.x, a.y + b.y);
}
// allow int2 - operator
DH_INLINE int2 operator-(const int2 &a, const int2 &b) {
    return make_int2(a.x - b.x, a.y - b.y);
}

// allow dividing float2 by float
DH_INLINE float2 operator/(const float2 &a, float s) {
    return make_float2(a.x / s, a.y / s);
}

DH_INLINE float2 &operator/=(float2 &a, float s) {
    a.x /= s;
    a.y /= s;
    return a;
}

// allow multiplying float2 by float
DH_INLINE float2 operator*(const float2 &a, float s) {
    return make_float2(a.x * s, a.y * s);
}

DH_INLINE float2 operator*(float s, const float2 &a) {
    return make_float2(s * a.x, s * a.y);
}

DH_INLINE float2 &operator*=(float2 &a, float s) {
    a.x *= s;
    a.y *= s;
    return a;
}

#pragma endregion
