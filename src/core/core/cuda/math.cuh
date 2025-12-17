/*

cuda math functions

*/
#pragma once
#include <cuda_runtime.h>

#define D_INLINE __device__ __forceinline__
#define DH_INLINE __device__ __host__ __forceinline__

namespace core::cuda::math {

#pragma region CONSTANTS

// // device code can't see constexpr?? seems it might
constexpr float SQRT2 = 1.4142135623730950488f;      // root of 2
constexpr float INV_SQRT2 = 0.70710678118654752440f; // inverse root of 2
constexpr float PI = 3.14159265358979323846f;
constexpr float GOLDEN_RATIO = 1.6180339887498948482f;

#pragma endregion

#pragma region LERP

// Generic scalar lerp
template <typename T>
DH_INLINE T lerp(T a, T b, T fade) {
    return a * (T(1) - fade) + b * fade;
}

// Overload for float2
DH_INLINE float2 lerp(float2 a, float2 b, float fade) {
    return make_float2(
        a.x * (1.0f - fade) + b.x * fade,
        a.y * (1.0f - fade) + b.y * fade);
}

// Overload for float3
DH_INLINE float3 lerp(float3 a, float3 b, float fade) {
    return make_float3(
        a.x * (1.0f - fade) + b.x * fade,
        a.y * (1.0f - fade) + b.y * fade,
        a.z * (1.0f - fade) + b.z * fade);
}

// Overload for float4
DH_INLINE float4 lerp(float4 a, float4 b, float fade) {
    return make_float4(
        a.x * (1.0f - fade) + b.x * fade,
        a.y * (1.0f - fade) + b.y * fade,
        a.z * (1.0f - fade) + b.z * fade,
        a.w * (1.0f - fade) + b.w * fade);
}

#pragma endregion

#pragma region IDX // pos to idx formulae

// pos to idx, note if we have layers (usually colour channels) we multiply the idx, then (R,G,B) = (idx+0, idx+1, idx+2)
DH_INLINE int pos_to_idx(int2 pos, int map_width, int layers = 1) {
    return (pos.y * map_width + pos.x) * layers;
}

// pos to idx formula
DH_INLINE int pos_to_idx(int x, int y, int map_width, int layers = 1) {
    return (y * map_width + x) * layers;
}

#pragma endregion

#pragma region POSMOD // wrapping coordinates

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

#pragma region CLAMP // clamp templates

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

#pragma region WRAP_OR_CLAMP // sampling image coordinates with wrap or clamp

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

#pragma region VECTOR_OPS // length, dot, cross, normalize

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

#pragma region HASH // MurmurHash3 hash

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

#pragma region SMOOTHING_AND_SATURATION // soft saturate

// soft ceiling to avoid clipping artifacts where sharpness controls how sharp or gentle the saturation curve is.
DH_INLINE float soft_saturate(float value, float ceiling, float sharpness = 1.0) {
    return ceiling * tanh((value / ceiling) * sharpness);
}

#pragma endregion

#pragma region SLOPES // calculate slope vectors, however in practise we hardcode this as we need jitter

// Master implementation: takes up to 3 maps, any nullptr is ignored
DH_INLINE float2 calculate_slope_vector(
    const float *__restrict__ height_map1,
    const float *__restrict__ height_map2,
    const float *__restrict__ height_map3,
    const int2 map_size,
    const int2 pos,
    const bool wrap = true,
    const float jitter = 0.0,
    const int step = 0,        // used by jitter, needs to be a different value each step
    const int jitter_mode = 0, // 0 is economical and less accurate
    const float scale = 1.0f,  // larger scale will make slopes less steep
    const int jitter_seed = 1234

) {

    int xp = wrap_or_clamp_index(pos.x + 1, map_size.x, wrap); // x + 1
    int xn = wrap_or_clamp_index(pos.x - 1, map_size.x, wrap); // x - 1
    int yp = wrap_or_clamp_index(pos.y + 1, map_size.y, wrap); // y + 1
    int yn = wrap_or_clamp_index(pos.y - 1, map_size.y, wrap); // y - 1

    int xp_idx = pos.y * map_size.x + xp; // {+1,0}
    int xn_idx = pos.y * map_size.x + xn; // {-1,0}
    int yp_idx = yp * map_size.x + pos.x; // {0,+1}
    int yn_idx = yn * map_size.x + pos.x; // {0,-1}

    float xp_height = height_map1[xp_idx]; // x+ height
    float yp_height = height_map1[yp_idx]; // y+ height
    float xn_height = height_map1[xn_idx]; // x- height
    float yn_height = height_map1[yn_idx]; // y- height

    if (height_map2) {
        xp_height += height_map2[xp_idx];
        yp_height += height_map2[yp_idx];
        xn_height += height_map2[xn_idx];
        yn_height += height_map2[yn_idx];
    }
    if (height_map3) {
        xp_height += height_map3[xp_idx];
        yp_height += height_map3[yp_idx];
        xn_height += height_map3[xn_idx];
        yn_height += height_map3[yn_idx];
    }

    // scale
    xp_height /= scale;
    yp_height /= scale;
    xn_height /= scale;
    yn_height /= scale;

    // ================================================================
    // [Jitter]
    // ----------------------------------------------------------------
    if (jitter > 0.0f) {
        switch (jitter_mode) {
        case 0: { // cheaper, reuses one hash, lower quality random shouldn't be a problem over frames
            uint32_t h = hash_uint(pos.x, pos.y, step, jitter_seed);
            xp_height += hash_to_4randf(h, 0) * jitter;
            yp_height += hash_to_4randf(h, 1) * jitter;
            xn_height += hash_to_4randf(h, 2) * jitter;
            yn_height += hash_to_4randf(h, 3) * jitter;
            break;
        }
        case 1: { // uses 4 hashes, technically better random
            xp_height += hash_float_signed(pos.x, pos.y, step, jitter_seed + 0) * jitter;
            yp_height += hash_float_signed(pos.x, pos.y, step, jitter_seed + 1) * jitter;
            xn_height += hash_float_signed(pos.x, pos.y, step, jitter_seed + 2) * jitter;
            yn_height += hash_float_signed(pos.x, pos.y, step, jitter_seed + 3) * jitter;
            break;
        }
        }
    }

    return float2{xn_height - xp_height, yn_height - yp_height};
}

// Overload: one map
DH_INLINE float2 calculate_slope_vector(
    const float *__restrict__ height_map1,
    const int2 map_size,
    const int2 pos,
    const bool wrap = true) {
    return calculate_slope_vector(height_map1, nullptr, nullptr, map_size, pos, wrap);
}

// Overload: two maps
DH_INLINE float2 calculate_slope_vector(
    const float *__restrict__ height_map1,
    const float *__restrict__ height_map2,
    const int2 map_size,
    const int2 pos,
    const bool wrap = true) {
    return calculate_slope_vector(height_map1, height_map2, nullptr, map_size, pos, wrap);
}


// can't include here, would need cu file

// // calculate slope vectors kernel
// __global__ void calculate_slope_vectors_kernel(
//     const float *__restrict__ height_map1, // heightmap (required)
//     const float *__restrict__ height_map2, // or null
//     const float *__restrict__ height_map3, // or null
//     float2 *__restrict__ slope_vectors_out,
//     const int2 map_size,
//     const bool wrap = true,    // wrap coordinates
//     const float jitter = 0.0f, // if > 0.0 apply jitter
//     const int step = 0,        // used by jitter, needs to be a different value each step
//     const int jitter_mode = 0, // 0 is economical and less accurate
//     const float scale = 1.0f,  // larger scale will make slopes less steep
//     const int jitter_seed = 1234) {
//     // ================================================================
//     int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
//     if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
//         return;
//     int idx = pos_to_idx(pos, map_size.x);
//     // ================================================================
//     float2 slope_vector = calculate_slope_vector(height_map1, height_map2, height_map3, map_size, pos, wrap, jitter, step, jitter_mode, scale, jitter_seed);
//     slope_vectors_out[idx] = slope_vector;
// }

// calculate slope vectors kernel
// __global__ void calculate_slope_vectors_kernel(
//     const float *__restrict__ height_map1, // heightmap (required)
//     const float *__restrict__ height_map2, // or null
//     const float *__restrict__ height_map3, // or null
//     float *__restrict__ slope_vectors_out, // must be double size of height_maps (interleaved the vectors)
//     const int2 map_size,
//     const bool wrap = true,    // wrap coordinates
//     const float jitter = 0.0f, // if > 0.0 apply jitter
//     const int step = 0,        // used by jitter, needs to be a different value each step
//     const int jitter_mode = 0, // 0 is economical and less accurate
//     const float scale = 1.0f,  // larger scale will make slopes less steep
//     const int jitter_seed = 1234) {
//     // ================================================================
//     int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
//     if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
//         return;
//     int idx = pos_to_idx(pos, map_size.x) * 2;
//     // ================================================================
//     float2 slope_vector = calculate_slope_vector(height_map1, height_map2, height_map3, map_size, pos, wrap, jitter, step, jitter_mode, scale, jitter_seed);
//     slope_vectors_out[idx] = slope_vector.x;
//     slope_vectors_out[idx + 1] = slope_vector.y;
// }

#pragma endregion

} 

#pragma region EQUALITY_OPERATORS // equality for vector 2D and 3D vector types

// 2-component
#define DEFINE_VEC2_EQ_OPS(TYPE)                              \
    DH_INLINE bool operator==(const TYPE &a, const TYPE &b) { \
        return (a.x == b.x) && (a.y == b.y);                  \
    }                                                         \
    DH_INLINE bool operator!=(const TYPE &a, const TYPE &b) { \
        return (a.x != b.x) || (a.y != b.y);                  \
    }

// 3-component
#define DEFINE_VEC3_EQ_OPS(TYPE)                              \
    DH_INLINE bool operator==(const TYPE &a, const TYPE &b) { \
        return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);  \
    }                                                         \
    DH_INLINE bool operator!=(const TYPE &a, const TYPE &b) { \
        return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);  \
    }

// 4-component
#define DEFINE_VEC4_EQ_OPS(TYPE)                                             \
    DH_INLINE bool operator==(const TYPE &a, const TYPE &b) {                \
        return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w); \
    }                                                                        \
    DH_INLINE bool operator!=(const TYPE &a, const TYPE &b) {                \
        return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w); \
    }

// ints
DEFINE_VEC2_EQ_OPS(int2)
DEFINE_VEC3_EQ_OPS(int3)
DEFINE_VEC4_EQ_OPS(int4)

// floats
DEFINE_VEC2_EQ_OPS(float2)
DEFINE_VEC3_EQ_OPS(float3)
DEFINE_VEC4_EQ_OPS(float4)

// doubles
DEFINE_VEC2_EQ_OPS(double2)
DEFINE_VEC3_EQ_OPS(double3)
DEFINE_VEC4_EQ_OPS(double4)

// unsigned ints
DEFINE_VEC2_EQ_OPS(uint2)
DEFINE_VEC3_EQ_OPS(uint3)
DEFINE_VEC4_EQ_OPS(uint4)

// note register preassure is not reduced with shorts and chars, but the array's can be smaller

// // signed chars (8 bit)
// DEFINE_VEC2_EQ_OPS(char2)
// DEFINE_VEC3_EQ_OPS(char3)
// DEFINE_VEC4_EQ_OPS(char4)

// // unsigned chars (8 bit)
// DEFINE_VEC2_EQ_OPS(uchar2)
// DEFINE_VEC3_EQ_OPS(uchar3)
// DEFINE_VEC4_EQ_OPS(uchar4)

// // signed shorts (16 bit)
// DEFINE_VEC2_EQ_OPS(short2)
// DEFINE_VEC3_EQ_OPS(short3)
// DEFINE_VEC4_EQ_OPS(short4)

// // unsigned shorts (16 bit)
// DEFINE_VEC2_EQ_OPS(ushort2)
// DEFINE_VEC3_EQ_OPS(ushort3)
// DEFINE_VEC4_EQ_OPS(ushort4)

// // signed longs (32‑bit on CUDA)
// DEFINE_VEC2_EQ_OPS(long2)
// DEFINE_VEC3_EQ_OPS(long3)
// DEFINE_VEC4_EQ_OPS(long4)

// // unsigned longs
// DEFINE_VEC2_EQ_OPS(ulong2)
// DEFINE_VEC3_EQ_OPS(ulong3)
// DEFINE_VEC4_EQ_OPS(ulong4)

#pragma endregion

#pragma region OPERATORS

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

// #undef D_INLINE
// #undef DH_INLINE