/*

store some kernels

*/
// #pragma once

namespace TEMPLATE_NAMESPACE {

#pragma region HASH

// Modern integer hash (based on MurmurHash3 finalizer)
__device__ __forceinline__ int hash_int(int x, int y, int z, int seed) {
    int n = x + y * 374761393 + z * 668265263 + seed * 1274126177;

    n ^= n >> 16;
    n *= 0x85ebca6b;
    n ^= n >> 13;
    n *= 0xc2b2ae35;
    n ^= n >> 16;

    return n & 0x7fffffff; // Keep positive for compatibility
}

// Hash returning float in [0,1)
__device__ __forceinline__ float hash_float(int x, int y, int z, int seed) {
    int h = hash_int(x, y, z, seed);
    // Scale to [0,1). Use 1.0f / 2147483648.0f (2^31)
    return static_cast<float>(h) * (1.0f / 2147483648.0f);
}

// If you want [-1,1] range:
__device__ __forceinline__ float hash_float_signed(int x, int y, int z, int seed) {
    int h = hash_int(x, y, z, seed);
    return static_cast<float>(h) * (2.0f / 2147483648.0f) - 1.0f;
}

#pragma endregion

#pragma region HELPERS

constexpr float SQRT2 = 1.4142135623730950488f;

// 8 offsets with the opposites in pairs
__device__ __constant__ int2 offsets[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
__device__ __constant__ float offset_distances[8] = {1.0f, 1.0f, 1.0f, 1.0f, SQRT2, SQRT2, SQRT2, SQRT2};
__device__ __constant__ int opposite_offset_refs[8] = {1, 0, 3, 2, 5, 4, 7, 6};

// tested posmod for wrapping map coordinates
__device__ __forceinline__ int posmod(int i, int mod) {
    int result = i % mod;
    return result < 0 ? result + mod : result;
}

// wrap or clamp for map coordinates, note the clamp is range-1
__device__ __forceinline__ int wrap_or_clamp(int i, int range, bool wrap) {
    if (wrap) {
        return posmod(i, range);
    } else {
        return i < 0 ? 0 : (i >= range ? range - 1 : i);
    }
}

__device__ __forceinline__ int clampi(int value, int minimum, int maximum) {
    return min(max(value, minimum), maximum);
}

__device__ __forceinline__ float clampf(float value, float minimum, float maximum) {
    return fminf(fmaxf(value, minimum), maximum);
}

#pragma endregion

} // namespace TEMPLATE_NAMESPACE
