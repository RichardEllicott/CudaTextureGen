/*

cuda math functions

*/
#pragma once
// #include "math_constants.cuh"
// #include "math_random.cuh"
#include <cstdint> // uint32_t
#include <cuda_runtime.h>
// #include <math_functions.h>  // for fmaxf, fminf, etc.

#define D_INLINE __device__ __forceinline__
#define DH_INLINE __device__ __host__ __forceinline__

namespace core::cuda::math {

#pragma region CONSTANTS

constexpr double SQRT2 = 1.4142135623730950488;     // square root of 2
constexpr double INV_SQRT2 = 0.7071067811865475244; // inverse square root of 2
constexpr double PI = 3.14159265358979323846;       // π ratio
constexpr double TAU = PI * 2.0;                    // 2π

constexpr uint32_t GOLDEN_RATIO_CONST = 0x9E3779B9u; // 32‑bit golden ratio constant (Knuth / SplitMix / xxHash)
constexpr uint32_t MURMUR3_C1 = 0x85EBCA6Bu;         // MurmurHash3 avalanche constant C1
constexpr uint32_t MURMUR3_C2 = 0xC2B2AE35u;         // MurmurHash3 avalanche constant C2

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

// Notes
// 374761393u	0x165667B1	Prime5	xxHash32	Mix low bits, small inputs
// 668265263u	0x27D4EB2F	Prime4	xxHash32	Secondary avalanche
// 1274126177u	0x4CF5AD43	Prime3	xxHash32	Seed mixing

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

#pragma region NORMAL_DISTRIBUTION

// Box–Muller from two random floats [0,1]
DH_INLINE float2 normal_vector2(float r1, float r2) {

    r1 = max(r1, 1e-7f); // Guard against log(0)
    float r = sqrt(-2.0f * log(r1));
    float theta = r2 * TAU;

    float s, c;
    sincos(theta, &s, &c); // faster than seperate sin and cos
    return make_float2(r * c, r * s);
}

// // Box–Muller 2D normal (two internal hashes)
DH_INLINE float2 normal_vector2(int x, int y, int z, int seed) {

    uint32_t h1 = hash_uint(x, y, z, seed);

    // reduce correlation for second hash with avalanche constants
    uint32_t h2 = hash_uint(
        h1,
        h1 ^ GOLDEN_RATIO_CONST, // golden ratio constant
        seed ^ MURMUR3_C1,       // Murmur3 C1
        MURMUR3_C2               // Murmur3 C2
    );

    constexpr float INV_U32 = 1.0f / 4294967296.0f; // 1/(2^32)

    float r1 = h1 * INV_U32;
    float r2 = h2 * INV_U32;

    return normal_vector2(r1, r2); // your Box–Muller version
}

// faster Box–Muller from just one 32 bit hash (16 bits for each part)
DH_INLINE float2 normal_vector2_fast(uint32_t h) {
    uint16_t lo = static_cast<uint16_t>(h);       // lower 16 bits
    uint16_t hi = static_cast<uint16_t>(h >> 16); // upper 16 bits
    float r1 = lo / 65536.0f;                     // => [0,1]
    float r2 = hi / 65536.0f;                     //=> [0,1]
    return normal_vector2(r1, r2);
}

// faster Box–Muller from just one 32 bit hash (16 bits for each part)
DH_INLINE float2 normal_vector2_fast(int x, int y, int z, int seed) {
    return normal_vector2_fast(hash_uint(x, y, z, seed));
}

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

// pos to idx shortcut
D_INLINE int pos_to_idx(const int2 pos, const int map_width) {
    return pos.y * map_width + pos.x;
}

// pos to idx shortcut
D_INLINE int pos_to_idx(const int2 pos, const int2 map_size) {
    return pos.y * map_size.x + pos.x;
}

// pos to idx formula
D_INLINE int pos_to_idx(const int x, const int y, const int map_width) {
    return y * map_width + x;
}

// get position 1D
D_INLINE int global_thread_pos1() {
    return int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
}

// get position 2D
D_INLINE int2 global_thread_pos2() {
    return make_int2(
        int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x),
        int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y));
}

// get position 3D
D_INLINE int3 global_thread_pos3() {
    return make_int3(
        int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x),
        int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y),
        int(blockIdx.z) * int(blockDim.z) + int(threadIdx.z));
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

// float2

// length of float2 vector
DH_INLINE float length(const float2 &vector) {
    return sqrt(vector.x * vector.x + vector.y * vector.y);
}

// normalize float2 vector to length of 1.0
DH_INLINE float2 normalize(const float2 &v) {
    float len = sqrtf(v.x * v.x + v.y * v.y);
    return (len > 1e-6f) ? make_float2(v.x / len, v.y / len) : make_float2(0.0f, 0.0f);
}

// dot product of two float2's
DH_INLINE float dot(const float2 &a, const float2 &b) {
    return a.x * b.x + a.y * b.y;
}

// 2D cross product (returns scalar z-component)
DH_INLINE float cross(const float2 &a, const float2 &b) {
    return a.x * b.y - a.y * b.x;
}

// float3

// length of float3 vector
DH_INLINE float length(const float3 &v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// normalize float3 vector to length of 1.0
DH_INLINE float3 normalize(const float3 &v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return (len > 1e-6f) ? make_float3(v.x / len, v.y / len, v.z / len)
                         : make_float3(0.0f, 0.0f, 0.0f);
}

// dot product of two float3's
DH_INLINE float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product of two float3's (returns a float3)
DH_INLINE float3 cross(const float3 &a, const float3 &b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
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

// float jitter = 1.0f;
// int step = 0;
// bool wrap = true;
// int jitter_mode = 0;
// float scale = 1.0f;
// int jitter_seed = 1234;

// float2 slope_vector2 = core::cuda::math::calculate_slope_vector(
//     height_map, water_map, nullptr, map_size, pos, wrap, jitter, step, jitter_mode, scale, jitter_seed);

// // Overload: one map
// DH_INLINE float2 calculate_slope_vector(
//     const float *__restrict__ height_map1,
//     const int2 map_size,
//     const int2 pos,
//     const bool wrap = true) {
//     return calculate_slope_vector(height_map1, nullptr, nullptr, map_size, pos, wrap);
// }

// // Overload: two maps
// DH_INLINE float2 calculate_slope_vector(
//     const float *__restrict__ height_map1,
//     const float *__restrict__ height_map2,
//     const int2 map_size,
//     const int2 pos,
//     const bool wrap = true) {
//     return calculate_slope_vector(height_map1, height_map2, nullptr, map_size, pos, wrap);
// }

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

#pragma region SMOOTHSTEP

// quintic smoothstep
// aka Perlin’s fade function
// Creates an S-curve (sigmoid-like shape)
__device__ __forceinline__ float quintic_smoothstep(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

#pragma endregion

} // namespace core::cuda::math

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

#pragma region ADDITION_SUBTRACTION

// 2-component
#define DEFINE_VEC2_ADD_SUB_OPS(TYPE)                        \
    DH_INLINE TYPE operator+(const TYPE &a, const TYPE &b) { \
        return {a.x + b.x, a.y + b.y};                       \
    }                                                        \
    DH_INLINE TYPE &operator+=(TYPE &a, const TYPE &b) {     \
        a.x += b.x;                                          \
        a.y += b.y;                                          \
        return a;                                            \
    }                                                        \
    DH_INLINE TYPE operator-(const TYPE &a, const TYPE &b) { \
        return {a.x - b.x, a.y - b.y};                       \
    }                                                        \
    DH_INLINE TYPE &operator-=(TYPE &a, const TYPE &b) {     \
        a.x -= b.x;                                          \
        a.y -= b.y;                                          \
        return a;                                            \
    }

// 3-component
#define DEFINE_VEC3_ADD_SUB_OPS(TYPE)                        \
    DH_INLINE TYPE operator+(const TYPE &a, const TYPE &b) { \
        return {a.x + b.x, a.y + b.y, a.z + b.z};            \
    }                                                        \
    DH_INLINE TYPE &operator+=(TYPE &a, const TYPE &b) {     \
        a.x += b.x;                                          \
        a.y += b.y;                                          \
        a.z += b.z;                                          \
        return a;                                            \
    }                                                        \
    DH_INLINE TYPE operator-(const TYPE &a, const TYPE &b) { \
        return {a.x - b.x, a.y - b.y, a.z - b.z};            \
    }                                                        \
    DH_INLINE TYPE &operator-=(TYPE &a, const TYPE &b) {     \
        a.x -= b.x;                                          \
        a.y -= b.y;                                          \
        a.z -= b.z;                                          \
        return a;                                            \
    }

// 4-component
#define DEFINE_VEC4_ADD_SUB_OPS(TYPE)                        \
    DH_INLINE TYPE operator+(const TYPE &a, const TYPE &b) { \
        return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; \
    }                                                        \
    DH_INLINE TYPE &operator+=(TYPE &a, const TYPE &b) {     \
        a.x += b.x;                                          \
        a.y += b.y;                                          \
        a.z += b.z;                                          \
        a.w += b.w;                                          \
        return a;                                            \
    }                                                        \
    DH_INLINE TYPE operator-(const TYPE &a, const TYPE &b) { \
        return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; \
    }                                                        \
    DH_INLINE TYPE &operator-=(TYPE &a, const TYPE &b) {     \
        a.x -= b.x;                                          \
        a.y -= b.y;                                          \
        a.z -= b.z;                                          \
        a.w -= b.w;                                          \
        return a;                                            \
    }

// ints
DEFINE_VEC2_ADD_SUB_OPS(int2)
DEFINE_VEC3_ADD_SUB_OPS(int3)
DEFINE_VEC4_ADD_SUB_OPS(int4)

// floats
DEFINE_VEC2_ADD_SUB_OPS(float2)
DEFINE_VEC3_ADD_SUB_OPS(float3)
DEFINE_VEC4_ADD_SUB_OPS(float4)

// doubles
DEFINE_VEC2_ADD_SUB_OPS(double2)
DEFINE_VEC3_ADD_SUB_OPS(double3)
DEFINE_VEC4_ADD_SUB_OPS(double4)

// unsigned ints (broken??)
// DEFINE_VEC2_ADD_SUB_OPS(uint2)
// DEFINE_VEC3_ADD_SUB_OPS(uint3)
// DEFINE_VEC4_ADD_SUB_OPS(uint4)

#undef DEFINE_VEC2_ADD_SUB_OPS
#undef DEFINE_VEC3_ADD_SUB_OPS
#undef DEFINE_VEC4_ADD_SUB_OPS

#pragma endregion

#pragma region SCALAR_OPERATORS // allow [*=, /=] OR [*, /] for vectors, with a scalar on the right

// 2-component
#define DEFINE_VEC2_SCALAR_OPS(TYPE)                   \
    DH_INLINE TYPE operator*(const TYPE &a, float s) { \
        return {a.x * s, a.y * s};                     \
    }                                                  \
    DH_INLINE TYPE &operator*=(TYPE &a, float s) {     \
        a.x *= s;                                      \
        a.y *= s;                                      \
        return a;                                      \
    }                                                  \
    DH_INLINE TYPE operator/(const TYPE &a, float s) { \
        return {a.x / s, a.y / s};                     \
    }                                                  \
    DH_INLINE TYPE &operator/=(TYPE &a, float s) {     \
        a.x /= s;                                      \
        a.y /= s;                                      \
        return a;                                      \
    }

// 3-component
#define DEFINE_VEC3_SCALAR_OPS(TYPE)                   \
    DH_INLINE TYPE operator*(const TYPE &a, float s) { \
        return {a.x * s, a.y * s, a.z * s};            \
    }                                                  \
    DH_INLINE TYPE &operator*=(TYPE &a, float s) {     \
        a.x *= s;                                      \
        a.y *= s;                                      \
        a.z *= s;                                      \
        return a;                                      \
    }                                                  \
    DH_INLINE TYPE operator/(const TYPE &a, float s) { \
        return {a.x / s, a.y / s, a.z / s};            \
    }                                                  \
    DH_INLINE TYPE &operator/=(TYPE &a, float s) {     \
        a.x /= s;                                      \
        a.y /= s;                                      \
        a.z /= s;                                      \
        return a;                                      \
    }

// 4-component
#define DEFINE_VEC4_SCALAR_OPS(TYPE)                   \
    DH_INLINE TYPE operator*(const TYPE &a, float s) { \
        return {a.x * s, a.y * s, a.z * s, a.w * s};   \
    }                                                  \
    DH_INLINE TYPE &operator*=(TYPE &a, float s) {     \
        a.x *= s;                                      \
        a.y *= s;                                      \
        a.z *= s;                                      \
        a.w *= s;                                      \
        return a;                                      \
    }                                                  \
    DH_INLINE TYPE operator/(const TYPE &a, float s) { \
        return {a.x / s, a.y / s, a.z / s, a.w / s};   \
    }                                                  \
    DH_INLINE TYPE &operator/=(TYPE &a, float s) {     \
        a.x /= s;                                      \
        a.y /= s;                                      \
        a.z /= s;                                      \
        a.w /= s;                                      \
        return a;                                      \
    }

// floats
DEFINE_VEC2_SCALAR_OPS(float2)
DEFINE_VEC3_SCALAR_OPS(float3)
DEFINE_VEC4_SCALAR_OPS(float4)

// doubles
DEFINE_VEC2_SCALAR_OPS(double2)
DEFINE_VEC3_SCALAR_OPS(double3)

// DEFINE_VEC4_SCALAR_OPS(double4)

#undef DEFINE_VEC2_SCALAR_OPS
#undef DEFINE_VEC3_SCALAR_OPS
#undef DEFINE_VEC4_SCALAR_OPS

#pragma endregion

// #undef D_INLINE
// #undef DH_INLINE