/*

cuda math functions

*/
#pragma once

#pragma region POSMOD

// tested posmod for wrapping map coordinates
__device__ __forceinline__ int posmod(int i, int mod) {
    int result = i % mod;
    return result < 0 ? result + mod : result;
}
// float version, haven't checked yet
__device__ __forceinline__ float posmod(float x, float mod) {
    float result = fmodf(x, mod); // remainder in (-mod, mod)
    return result < 0.0f ? result + mod : result;
}

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

// allow int2 + operator
__device__ __host__ inline int2 operator+(const int2 &a, const int2 &b) {
    return make_int2(a.x + b.x, a.y + b.y);
}
// allow int2 - operator
__device__ __host__ inline int2 operator-(const int2 &a, const int2 &b) {
    return make_int2(a.x - b.x, a.y - b.y);
}
// allow int2 == operator
__device__ __host__ inline bool operator==(const int2 &a, const int2 &b) {
    return (a.x == b.x) && (a.y == b.y);
}
// allow int2 != operator
__device__ __host__ inline bool operator!=(const int2 &a, const int2 &b) {
    return (a.x != b.x) || (a.y != b.y);
}
