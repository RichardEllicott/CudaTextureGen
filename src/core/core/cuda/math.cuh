/*

cuda math functions, main library

*/
#pragma once

#include "math_fast.cuh" // fast math intrinsics
#include <cstdint>       // uint32_t
#include <cuda_runtime.h>

#pragma region DEFINES

#define D_INLINE __device__ __forceinline__           // device only functions
#define DH_INLINE __device__ __host__ __forceinline__ // device and host functions

// define DH_CONST different for host and device
#ifdef __CUDACC__
#define DH_CONST __device__ __constant__ const
#else
#define DH_CONST constexpr
#endif

#pragma endregion

namespace core::cuda::math {

#pragma region CONSTANTS

// ⚠️ adding inline to constexpr can suppress compiler warning
// ⚠️ can't use "std::sqrt(2.0)" in constexpr due to CUDA (so using literals)

constexpr float SQRT2 = 1.4142135623730950488;     // square root of 2
constexpr float INV_SQRT2 = 0.7071067811865475244; // inverse square root of 2
constexpr float PI = 3.14159265358979323846;       // π ratio
constexpr float TAU = PI * 2.0f;

#pragma region MAGIC_HASH_NUMBERS

constexpr uint32_t GOLDEN_RATIO_CONST = 0x9E3779B9u; // 32‑bit golden ratio constant (Knuth / SplitMix / xxHash)
constexpr uint32_t MURMUR3_C1 = 0x85EBCA6Bu;         // MurmurHash3 avalanche constant C1
constexpr uint32_t MURMUR3_C2 = 0xC2B2AE35u;         // MurmurHash3 avalanche constant C2

constexpr uint32_t XXH_PRIME32_3 = 1274126177u; // xxHash32 Seed mixing
constexpr uint32_t XXH_PRIME32_4 = 668265263u;  // xxHash32 Secondary avalanche
constexpr uint32_t XXH_PRIME32_5 = 374761393u;  // xxHash32 Mix low bits, small inputs

// numbers used to multiply a hash to floats
constexpr float INV_U32 = 0x1p-32f; // exactly 2^-32
constexpr float INV_U31 = 0x1p-31f; // exactly 2^-31

#pragma endregion

#pragma region GRID

// standard grid offsets in standard order (opposites in pairs, first 4 are cardinal, second 4 are diagonal)
DH_CONST int2 GRID_OFFSETS_8[8] =
    {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

// opposite direction indexes
DH_CONST int GRID_OFFSETS_8_OPPOSITE_INDEX[8] =
    {1, 0, 3, 2, 5, 4, 7, 6};

// distance to neighbours
DH_CONST float GRID_OFFSETS_8_DISTANCES[8] =
    {1.0f, 1.0f, 1.0f, 1.0f, SQRT2, SQRT2, SQRT2, SQRT2};

// special vectors for creating a scaled dot product
DH_CONST float2 GRID_OFFSETS_8_DOTS[8] =
    {{1.0f, 0.0f}, {-1.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, -1.0f}, {0.5f, 0.5f}, {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}};

// the directions normalized (magnitude of 1.0)
DH_CONST float2 GRID_OFFSETS_8_NORMALIZED[8] =
    {{1.0f, 0.0f}, {-1.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, -1.0f}, {INV_SQRT2, INV_SQRT2}, {-INV_SQRT2, -INV_SQRT2}, {INV_SQRT2, -INV_SQRT2}, {-INV_SQRT2, INV_SQRT2}};

#pragma endregion

#pragma endregion

#pragma region CUDA_ONLY // these ones only compile correct from .cu files

#ifdef __CUDACC__

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
#pragma endregion

#pragma region HASH // MurmurHash3 hash

// // signed integer hash (based on MurmurHash3 finalizer)
DH_INLINE int hash_int(int x, int y, int z, int seed) {
    int n = x + y * XXH_PRIME32_5 + z * XXH_PRIME32_4 + seed * XXH_PRIME32_3;

    n ^= n >> 16;
    n *= MURMUR3_C1;
    n ^= n >> 13;
    n *= MURMUR3_C2;
    n ^= n >> 16;

    return n;
}

// integer hash (based on MurmurHash3 finalizer)
DH_INLINE uint32_t hash_uint(uint32_t x, uint32_t y, uint32_t z, uint32_t seed) {
    uint32_t n = x + y * XXH_PRIME32_5 + z * XXH_PRIME32_4 + seed * XXH_PRIME32_3;

    n ^= n >> 16;
    n *= MURMUR3_C1;
    n ^= n >> 13;
    n *= MURMUR3_C2;
    n ^= n >> 16;

    return n; // full 32-bit unsigned result
}

// float from [0,1]
DH_INLINE float hash_float(uint32_t hash) {
    return static_cast<float>(hash) * INV_U32; // Scale to [0,1]
}

// float from [0,1]
DH_INLINE float hash_float(uint32_t x, uint32_t y, uint32_t z, uint32_t seed) {
    return hash_float(hash_uint(x, y, z, seed));
}

// float from [-1,1]
DH_INLINE float hash_float_signed(int hash) {
    return static_cast<float>(hash) * INV_U31; // Scale to [-1,1).
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

// ================================================================================================================================
// Idea to use one hash for four floats with mixing
// gives high quality random at cheaper cost
// --------------------------------------------------------------------------------------------------------------------------------

// So four hashes = ~40–60 integer ops.

// On CPU, that’s noticeable.
// On GPU, that’s very noticeable because integer multiplies and shifts aren’t free.

// ✔ ~30 integer ops
// vs

// ❌ ~50 integer ops for four independent hashes
// That’s already a ~40% reduction.

// 🚀 1.5× to 2× faster
// for “expand one hash into four floats” vs “four independent hashes”.

// high entrophy cheap hashing??
DH_INLINE uint32_t hash_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

struct HashRng {
    uint32_t x;

    // output float [-1, 1]
    DH_INLINE float next_signed() {
        x = hash_mix(x);

#if defined(__CUDA_ARCH__) // most correct CUDA pattern for inside a DH_INLINE
        // Compiling for device
        return __int2float_rn(int32_t(x)) * INV_U31;
#else
        // Compiling for host
        return int32_t(x) * INV_U31;
#endif
    }

    DH_INLINE float next_unsigned() {
        x = hash_mix(x);
        return hash_float(x); // [0,1)
    }
};

// HashRng rng{mix(seed_base ^ thread_id)};
// float a = rng.next_signed();
// float b = rng.next_signed();
// float c = rng.next_unsigned();

// ================================================================================================================================

#pragma endregion

#pragma region NORMAL_DISTRIBUTION

// __sincosf	Fastest	Low	Best for your kernels
// __sinf / __cosf	Very fast	Low	Good if you only need one
// sincosf	Medium	High	What you use now
// sin / cos	Slow	Double	Avoid

// Box–Muller from two random floats [0,1]
// ⚠️ we have found faster sincos functions!!
DH_INLINE float2 normal_vector2(float r1, float r2) {

    r1 = fmaxf(r1, 1e-7f); // Guard against log(0)
    float r = sqrtf(-2.0f * fast_logf(r1));

    float theta = r2 * TAU;

    float s, c;
    // sincosf(theta, &s, &c); // faster than seperate sin and cos
    fast_sincosf(theta, &s, &c);

    return make_float2(r * c, r * s);
}

// // Marsaglia polar could be faster?
// // two random floats [0,1]
// DH_INLINE float2 marsaglia_normal_vector2(float r1, float r2) {
//     // Transform to [-1, 1]
//     float x = 2.0f * r1 - 1.0f;
//     float y = 2.0f * r2 - 1.0f;

//     float s = x * x + y * y; // pythagoras

//     // Rejection: extremely rare, but must be handled
//     // (You can push this to the caller if you want branch-free kernels)
//     if (s >= 1.0f || s <= 1e-12f) {
//         // Degenerate case: fall back to Box–Muller style
//         r1 = max(r1, 1e-7f);
//         float r = sqrtf(-2.0f * logf(r1));
//         float theta = r2 * PI * 2.0f;
//         float s2, c2;
//         sincosf(theta, &s2, &c2);
//         return make_float2(r * c2, r * s2);
//     }

//     float factor = sqrtf(-2.0f * logf(s) / s);
//     return make_float2(x * factor, y * factor);
// }

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

    // constexpr float INV_U32 = 1.0f / 4294967296.0f; // 1/(2^32)

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

#pragma region IDX // pos to idx formulae

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
DH_INLINE float length(float2 vector) {
    return sqrtf(vector.x * vector.x + vector.y * vector.y);
}

// length of float3 vector
DH_INLINE float length(float3 vector) {
    return sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

// // normalize float2 vector to length of 1.0
// DH_INLINE float2 normalize(float2 vector) {
//     float len = fast_sqrtf(vector.x * vector.x + vector.y * vector.y);
//     return (len > 1e-6f) ? make_float2(vector.x / len, vector.y / len) : make_float2(0.0f, 0.0f);
// }

// // normalize float2 vector to length of 1.0 (with fast reciprocal root)
DH_INLINE float2 normalize(float2 v) {
    float len2 = v.x * v.x + v.y * v.y;
    if (len2 > 1e-12f) {
        float inv = fast_rsqrtf(len2); // 1/sqrt(len2)
        return make_float2(v.x * inv, v.y * inv);
    }
    return make_float2(0.0f, 0.0f);
}

// // normalize float3 vector to length of 1.0
// DH_INLINE float3 normalize(float3 vector) {
//     float len = fast_sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
//     return (len > 1e-6f) ? make_float3(vector.x / len, vector.y / len, vector.z / len)
//                          : make_float3(0.0f, 0.0f, 0.0f);
// }

// normalize float3 vector to length of 1.0 (with fast reciprocal root)
DH_INLINE float3 normalize(float3 v) {
    float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len2 > 1e-12f) {               // squared epsilon
        float inv = fast_rsqrtf(len2); // 1/sqrt(len2)
        return make_float3(v.x * inv,
                           v.y * inv,
                           v.z * inv);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

// dot product of two float2's
DH_INLINE float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

// dot product of two float3's
DH_INLINE float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// 2D cross product (returns scalar z-component)
DH_INLINE float cross(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }

// cross product of two float3's (returns a float3)
DH_INLINE float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

#pragma endregion

#pragma region SMOOTHING_AND_SATURATION // soft saturate

// soft ceiling to avoid clipping artifacts where sharpness controls how sharp or gentle the saturation curve is.
// uses hyperbolic tan which means we will never reach the ceiling
// higher sharpness saturates quicker
DH_INLINE float soft_saturate(float value, float ceiling, float sharpness = 1.0f) {
    return ceiling * fast_tanhf((value / ceiling) * sharpness);
}

#pragma endregion

#pragma region SLOPES // calculate slope vectors, however in practise we hardcode this as we need jitter

// Master implementation: takes up to 3 maps, any nullptr is ignored
DH_INLINE float2 compute_slope_vector_OLD(
    const float *__restrict__ height_map1,
    const float *__restrict__ height_map2,
    const float *__restrict__ height_map3,
    const int2 map_size,
    const int2 pos,
    const bool wrap = true,
    const float jitter = 0.0f,
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

    // // scale
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

#pragma endregion

#pragma region SMOOTHSTEP

// quintic smoothstep
// aka Perlin’s fade function
// Creates an S-curve (sigmoid-like shape)
DH_INLINE float quintic_smoothstep(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

// cheaper than quintic, but less accurate
__device__ __forceinline__ float cubic_smoothstep(float t) { return t * t * (3 - 2 * t); }

#pragma endregion

} // namespace core::cuda::math
