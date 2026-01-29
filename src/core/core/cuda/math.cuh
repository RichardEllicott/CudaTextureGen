/*

cuda math functions, main library

*/
#pragma once

#include <cstdint> // uint32_t (required for gcc)
#include <cuda_runtime.h>

#include "core/cuda/cast.cuh"
#include "core/cuda/hash.cuh"
#include "core/cuda/math/constants.cuh"
#include "core/cuda/math/fast.cuh" // intrinsics
#include "core/cuda/math/operators.cuh"
#include "core/defines.h"

namespace core::cuda::math {

using namespace constants; // refactoring

#pragma region RADIANS

DH_INLINE float radians(float deg) {
    return deg * constants::DEG_TO_RAD;
}

DH_INLINE float degrees(float rad) {
    return rad * constants::RAD_TO_DEG;
}

DH_INLINE float2 radians(float2 deg) {
    return make_float2(
        radians(deg.x),
        radians(deg.y));
}

DH_INLINE float3 radians(float3 deg) {
    return make_float3(
        radians(deg.x),
        radians(deg.y),
        radians(deg.z));
}

DH_INLINE float4 radians(float4 deg) {
    return make_float4(
        radians(deg.x),
        radians(deg.y),
        radians(deg.z),
        radians(deg.w));
}
// --------------------------------------------------------------------------------------------------------------------------------
DH_INLINE float2 degrees(float2 rad) {
    return make_float2(
        degrees(rad.x),
        degrees(rad.y));
}

DH_INLINE float3 degrees(float3 rad) {
    return make_float3(
        degrees(rad.x),
        degrees(rad.y),
        degrees(rad.z));
}

DH_INLINE float4 degrees(float4 rad) {
    return make_float4(
        degrees(rad.x),
        degrees(rad.y),
        degrees(rad.z),
        degrees(rad.w));
}
// --------------------------------------------------------------------------------------------------------------------------------
#pragma endregion

#pragma region FLOOR

DH_INLINE float floor(float v) {
    return floorf(v);
}

DH_INLINE float2 floor(float2 v) {
    return make_float2(
        floor(v.x),
        floor(v.y));
}

DH_INLINE float3 floor(float3 v) {
    return make_float3(
        floor(v.x),
        floor(v.y),
        floor(v.z));
}

DH_INLINE float4 floor(float4 v) {
    return make_float4(
        floor(v.x),
        floor(v.y),
        floor(v.z),
        floor(v.w));
}

#pragma endregion

#pragma region ABS

// correct abs for cuda
DH_INLINE float abs(float v) {
    return fabsf(v);
}

// absolute value for 32‑bit signed integers (branchless)
DH_INLINE int abs(int v) {
    int mask = v >> 31;
    return (v + mask) ^ mask;
}

#pragma endregion

#pragma region LERP

// // Generic scalar lerp
// template <typename T>
// DH_INLINE T lerp(T a, T b, T fade) {
//     return a * (T(1) - fade) + b * fade;
// }

DH_INLINE float lerp(float a, float b, float fade) {
    return a * (1.0f - fade) + b * fade;
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

#pragma region HASH // MurmurHash3 hash

using namespace core::cuda::hash;

#pragma endregion

#pragma region NORMAL_DISTRIBUTION

// Box–Muller from two random floats [0,1]
DH_INLINE float2 normal_vector2(float r1, float r2) {

    r1 = fmaxf(r1, 1e-7f); // Guard against log(0)
    float r = sqrtf(-2.0f * fast::log(r1));

    float theta = r2 * constants::TAU;

    float s, c;
    fast::sincos(theta, &s, &c);

    return make_float2(r * c, r * s);
}

// // Box–Muller 2D normal (two internal hashes)
DH_INLINE float2 normal_vector2(int x, int y, int z, int seed) {

    uint32_t hash = hash_uint(x, y, z, seed); // get hash
    float r1 = hash_float(hash);              // get ∈[0,1]
    hash = hash_mix(hash);                    // mix hash
    float r2 = hash_float(hash);              // get ∈[0,1]

    return normal_vector2(r1, r2); // your Box–Muller version
}

// faster Box–Muller from just one 32 bit hash (16 bits for each part)
DH_INLINE float2 normal_vector2_fast(uint32_t h) {
    uint16_t lo = static_cast<uint16_t>(h);       // lower 16 bits
    uint16_t hi = static_cast<uint16_t>(h >> 16); // upper 16 bits
    float r1 = lo / 65536.0f;                     // => ∈[0,1]
    float r2 = hi / 65536.0f;                     //=> ∈[0,1]
    return normal_vector2(r1, r2);
}

// faster Box–Muller from just one 32 bit hash (16 bits for each part)
DH_INLINE float2 normal_vector2_fast(int x, int y, int z, int seed) {
    return normal_vector2_fast(hash_uint(x, y, z, seed));
}

#pragma endregion

#pragma region POSMOD // wrapping coordinates

// positive modulo for wrapping map coordinates (branchless)
// NOTE: 'mod' must be strictly positive. Negative moduli are undefined here.
DH_INLINE int posmod(int i, int mod) {
    int r = i % mod;
    return r + ((r >> 31) & mod);
}

// positive modulo for float
// NOTE: branchless version not determined to give any benefit compared to the fmodf cost
DH_INLINE float posmod(float x, float mod) {
    float result = fmodf(x, mod); // remainder in (-mod, mod)
    return result < 0.0f ? result + mod : result;
}

// --------------------------------------------------------------------------------------------------------------------------------

// positive modulo on int2
DH_INLINE int2 posmod(int2 pos, int2 mod) {
    return make_int2(posmod(pos.x, mod.x), posmod(pos.y, mod.y));
}

// positive modulo on int3
DH_INLINE int3 posmod(int3 pos, int3 mod) {
    return make_int3(posmod(pos.x, mod.x), posmod(pos.y, mod.y), posmod(pos.y, mod.y));
}

// positive modulo on float2
DH_INLINE float2 posmod(float2 pos, float2 mod) {
    return make_float2(posmod(pos.x, mod.x), posmod(pos.y, mod.y));
}

// positive modulo on float3
DH_INLINE float3 posmod(float3 pos, float3 mod) {
    return make_float3(posmod(pos.x, mod.x), posmod(pos.y, mod.y), posmod(pos.y, mod.y));
}

#pragma endregion

#pragma region HELPERS // pos to idx, calc grid, thread pos
// ================================================================================================================================

// pos to idx shortcut
DH_INLINE int pos_to_idx(int2 pos, int map_width) {
    return pos.y * map_width + pos.x;
}

// pos to idx shortcut
DH_INLINE int pos_to_idx(int2 pos, int2 map_size) {
    return pos.y * map_size.x + pos.x;
}

// pos to idx formula
DH_INLINE int pos_to_idx(int x, int y, int map_width) {
    return y * map_width + x;
}

// --------------------------------------------------------------------------------------------------------------------------------

#ifdef __CUDACC__ // only use when compiling with NVCC

// get position in 1D kernel
D_INLINE int global_thread_pos1() {
    return int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
}

// get position in 2D kernel
D_INLINE int2 global_thread_pos2() {
    return make_int2(
        int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x),
        int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y));
}

// get position in 3D kernel
D_INLINE int3 global_thread_pos3() {
    return make_int3(
        int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x),
        int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y),
        int(blockIdx.z) * int(blockDim.z) + int(threadIdx.z));
}

#endif

// --------------------------------------------------------------------------------------------------------------------------------

__host__ inline dim3 calculate_grid(int2 size, dim3 block = dim3(16, 16)) {
    return dim3(
        (size.x + block.x - 1) / block.x,
        (size.y + block.y - 1) / block.y);
}

// ================================================================================================================================

#pragma endregion

#pragma region MIN_MAX_CLAMP // clamp templates

// float min
DH_INLINE float min(float a, float b) {
    return fminf(a, b);
}

// float max
DH_INLINE float max(float a, float b) {
    return fmaxf(a, b);
}

// int min
DH_INLINE int min(int a, int b) {
    return (a < b) ? a : b;
}

// int max
DH_INLINE int max(int a, int b) {
    return (a > b) ? a : b;
}

// uint min
DH_INLINE uint32_t min(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

// uint max
DH_INLINE uint32_t max(uint32_t a, uint32_t b) {
    return (a > b) ? a : b;
}

// ----------------------------------------------------------------

// float clamp
DH_INLINE float clamp(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

// general clamp (likely would have been just as fast for float)
template <typename T>
DH_INLINE T clamp(T v, T min, T max) {
    return v < min ? min : (v > max ? max : v);
}

// clamp integer to an index, ie range 8 => [0, 7]
DH_INLINE int clamp_index(int i, int range) {
    return clamp(i, 0, range - 1);
}

// clamp int2 to an index, ie range 8 => [0, 7]
DH_INLINE int2 clamp_index(int2 pos, int2 range) {
    return make_int2(
        clamp_index(pos.x, range.x),
        clamp_index(pos.y, range.y));
}

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

#define USE_FAST_LENGTH
#ifdef USE_FAST_LENGTH

// length of float2 vector
DH_INLINE float length(float2 v) {
    float s = v.x * v.x + v.y * v.y;
    return s * fast::rsqrt(s);
}

// length of float3 vector
DH_INLINE float length(float3 v) {
    float s = v.x * v.x + v.y * v.y + v.z * v.z;
    return s * fast::rsqrt(s);
}

#else
// length of float2 vector
DH_INLINE float length(float2 vector) {
    return sqrtf(vector.x * vector.x + vector.y * vector.y);
}

// length of float3 vector
DH_INLINE float length(float3 vector) {
    return sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}
#endif
#undef USE_FAST_LENGTH

// // ----------------------------------------------------------------
// // MIGHT BE FASTER
// // length of float2 vector
// DH_INLINE __device__ float length(float2 v) {
//     float s = __fmaf_rn(v.y, v.y, v.x * v.x);
//     return __fsqrt_rn(s);
// }

// // length of float3 vector
// DH_INLINE __device__ float length(float3 v) {
//     float s = __fmaf_rn(v.z, v.z,
//               __fmaf_rn(v.y, v.y,
//                          v.x * v.x));
//     return __fsqrt_rn(s);
// }
// // ----------------------------------------------------------------

// length of int2 vector
DH_INLINE float length(int2 vector) {
    return length(core::cuda::cast::to_float2(vector));
}

// length of int3 vector
DH_INLINE float length(int3 vector) {
    return length(core::cuda::cast::to_float3(vector));
}

// normalize float2 vector to length of 1.0 (with fast reciprocal root)
DH_INLINE float2 normalize(float2 v) {
    float len2 = v.x * v.x + v.y * v.y;
    if (len2 > 1e-12f) {
        float inv = fast::rsqrt(len2); // 1/sqrt(len2)
        return make_float2(v.x * inv, v.y * inv);
    }
    return make_float2(0.0f, 0.0f);
}

// normalize float3 vector to length of 1.0 (with fast reciprocal root)
DH_INLINE float3 normalize(float3 v) {
    float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len2 > 1e-12f) {               // squared epsilon
        float inv = fast::rsqrt(len2); // 1/sqrt(len2)
        return make_float3(v.x * inv,
                           v.y * inv,
                           v.z * inv);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

// ----------------------------------------------------------------

// dot product of two float2's
DH_INLINE float dot(float2 a, float2 b) {
    // return a.x * b.x + a.y * b.y;
    return fast::fma(a.y, b.y, a.x * b.x);
}

// dot product of two float3's
DH_INLINE __device__ float dot(float3 a, float3 b) {
    return fast::fma(a.z, b.z,
                     fast::fma(a.y, b.y,
                               a.x * b.x));
}

// 2D cross product (returns scalar z-component)
DH_INLINE float cross(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }

// cross product of two float3's (returns a float3)
DH_INLINE float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

// ----------------------------------------------------------------

// rotate vector clockwise
DH_INLINE int2 rotate(int2 v, int quarter_turns = 1) {
    switch (posmod(quarter_turns, 4)) {
    case 1:
        return make_int2(-v.y, v.x); // 90°  (-y, x)
    case 2:
        return make_int2(-v.x, -v.y); // 180° (-x, -y)
    case 3:
        return make_int2(v.y, -v.x); // 270° (y, -x)
    }
    return v;
}

// rotate vector clockwise
DH_INLINE float2 rotate(float2 v, int quarter_turns = 1) {
    switch (posmod(quarter_turns, 4)) {
    case 1:
        return make_float2(-v.y, v.x); // 90°  (-y, x)
    case 2:
        return make_float2(-v.x, -v.y); // 180° (-x, -y)
    case 3:
        return make_float2(v.y, -v.x); // 270° (y, -x)
    }
    return v;
}

#pragma endregion

#pragma region SMOOTHING_AND_SATURATION // soft saturate

// soft ceiling to avoid clipping artifacts where sharpness controls how sharp or gentle the saturation curve is.
// uses hyperbolic tan which means we will never reach the ceiling
// higher sharpness saturates quicker
DH_INLINE float soft_saturate(float value, float ceiling, float sharpness = 1.0f) {
    return ceiling * fast::tanh((value / ceiling) * sharpness);
}

#pragma endregion

#pragma region SMOOTHSTEP

namespace smooth {

// Cubic smoothstep (C¹ continuous)
// top choice for Simplex Noise
DH_INLINE float cubic(float t) {
    return t * t * (3.0f - 2.0f * t);
}

// Quintic smoothstep (Perlin fade, C² continuous)
DH_INLINE float quintic(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Cosine interpolation — soft, wavy, analog feel
DH_INLINE float cosine(float t) {
    return 0.5f - 0.5f * fast::cos(t * constants::PI);
}

// Power-law smoothing — adjustable sharpness
// k > 1 sharpens, k < 1 softens
DH_INLINE float power(float t, float k) {
    return fast::pow(t, k);
}

// Hard sharpstep (binary)
DH_INLINE float sharp(float t) {
    return (t > 0.5f) ? 1.0f : 0.0f;
}

// Soft sharp (sharpened smoothstep)
DH_INLINE float ssharp(float t) {
    t = t * t; // exaggerate low end
    return t * (3.0f - 2.0f * t);
}

DH_INLINE float apply_smoothing(float t, int mode) {
    switch (mode) {
    case -1:
        return t; // pass
    case 0:       // cubic (this is likely best, lower quality than quintic but not visible)
        return cubic(t);
    case 1: // quintic
        return quintic(t);
    case 2: // cosine (not really appropiate, trying for style)
        return cosine(t);
    case 3: // sharp (for style)
        return sharp(t);
    case 4: // sharp (for style)
        return ssharp(t);
    case 5: // power (for style)
        return power(t, 2);
    default:
        return t; // pass
    }
}

} // namespace smooth

#pragma endregion

#pragma region FALLOFF_KERNELS

namespace kernel {

// gaussian distance based falloff, will be 1 at distance 0
DH_INLINE float gaussian(float distance, float sigma = 1.0f) {
    float a = distance / sigma;
    // return expf(-0.5f * a * a);
    return fast::exp(-0.5f * a * a);
}

// Cubic spline SPH kernel (compact support).
DH_INLINE float cubic_spline(float d, float h) {
    float q = d / h;
    if (q >= 2.0f) return 0.0f;

    if (q < 1.0f) {
        return 1.0f - 1.5f * q * q + 0.75f * q * q * q;
    } else {
        float t = 2.0f - q;
        return 0.25f * t * t * t;
    }
}

// Wendland C2 radial basis function (compactly supported).
DH_INLINE float wendland_c2(float d, float R) {
    if (d >= R) return 0.0f;
    float x = 1.0f - d / R;
    float x2 = x * x;
    return x2 * x2 * (4.0f * d / R + 1.0f);
}

} // namespace kernel

#pragma endregion

} // namespace core::cuda::math
