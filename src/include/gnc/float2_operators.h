/*

trying to get this in a .h file


*/
#pragma once

#include <cuda_runtime.h>

// Host-only operators for float2 (for use in CPU bindings code)
inline bool operator==(const float2& a, const float2& b) {
    return a.x == b.x && a.y == b.y;
}

inline bool operator!=(const float2& a, const float2& b) {
    return !(a == b);
}

// Add any other operators you need for CPU code
inline float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

