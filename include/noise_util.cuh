/*

functions for the noise generators, hash functions, dot products, gradients

*/
#pragma once

#define NOISE_UTIL_TRIG_HASH 1

namespace noise_util {

#pragma region UTILITY_MATH

// Creates an S-curve (sigmoid-like shape)
__device__ __forceinline__ float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

// simple interpolate
__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// positive modulo wrap (note it might be faster to concider other wrap methods depending on the situation)
__device__ __forceinline__ int posmod(int value, int mod) {
    int result = value % mod;
    return result < 0 ? result + mod : result;
}

// dot product for 3D
__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

#pragma endregion



#pragma region HASH_FUNCTIONS

__device__ __forceinline__ float trig_hash(float x, float y, float z, int seed) {
    float dot = x * 12.9898f + y * 78.233f + z * 37.719f + seed * 0.123f;
    return (sinf(dot) * 43758.5453f - floorf(sinf(dot) * 43758.5453f)) * 2.0f - 1.0f;
}

__device__ __forceinline__ int hash_int(int x, int y, int seed) {
    int n = x + y * 57 + seed * 131;
    n = (n << 13) ^ n;
    return (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
}

__device__ __forceinline__ float hash_scalar(int x, int y, int seed) {

#if NOISE_UTIL_TRIG_HASH == 0
    return 1.0f - hash_int(x, y, seed) / 1073741824.0f;
#elif NOISE_UTIL_TRIG_HASH == 1
    return trig_hash(x, y, 0, seed);
#endif
}

__device__ __forceinline__ int hash_int(int x, int y, int z, int seed) {
    int n = x + y * 57 + z * 113 + seed * 131;
    n = (n << 13) ^ n;
    return (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
}

__device__ __forceinline__ float hash_scalar(int x, int y, int z, int seed) {

#if NOISE_UTIL_TRIG_HASH == 0
    return 1.0f - hash_int(x, y, z, seed) / 1073741824.0f;
#elif NOISE_UTIL_TRIG_HASH == 1
    return trig_hash(x, y, z, seed);
#endif
}

#pragma endregion

#pragma region GRADIENT_GENERATORS

// 2D gradient for gradient noise (looks like simplex)
__device__ __forceinline__ float2 gradient(int x, int y, int seed) {
    float angle = (hash_int(x, y, seed) / 1073741824.0f) * 3.14159265f;
    return make_float2(cosf(angle), sinf(angle));
}

// 3D gradient for gradient noise (looks like simplex)
__device__ __forceinline__ float3 gradient(int x, int y, int z, int seed) {
    // Hash to get a pseudo-random angle and elevation
    float h1 = hash_int(x, y, z, seed) / 1073741824.0f; // range ~[0, 2]
    float h2 = hash_int(z, x, y, seed + 1337) / 1073741824.0f;

    // Convert to spherical coordinates
    float theta = h1 * 2.0f * 3.14159265f; // azimuthal angle
    float phi = h2 * 3.14159265f;          // polar angle

    float sin_phi = sinf(phi);
    return make_float3(
        cosf(theta) * sin_phi,
        sinf(theta) * sin_phi,
        cosf(phi));
}

#pragma endregion



} // namespace noise_util
