/*

host side maths

*/
#pragma once

#include <cstdint> // uint32_t
#include <vector>

#include "cuda_compat.h"

namespace core::math {


    // positive modulo for wrapping map coordinates (branchless)
// NOTE: 'mod' must be strictly positive. Negative moduli are undefined here.
inline int posmod(int i, int mod) {
    int r = i % mod;
    return r + ((r >> 31) & mod);
}

// positive modulo for float
// NOTE: branchless version not determined to give any benefit compared to the fmodf cost
inline float posmod(float x, float mod) {
    float result = fmodf(x, mod); // remainder in (-mod, mod)
    return result < 0.0f ? result + mod : result;
}



// rotate vector clockwise
inline int2 rotate(int2 vector, int quarter_turns = 1) {
    switch (posmod(quarter_turns, 4)) {
    default: return vector;                         // 0°
    case 1: return make_int2(-vector.y, vector.x);  // 90°
    case 2: return make_int2(-vector.x, -vector.y); // 180°
    case 3: return make_int2(vector.y, -vector.x);  // 270°
    }
}

// rotate vector clockwise
inline float2 rotate(float2 vector, int quarter_turns = 1) {
    switch (posmod(quarter_turns, 4)) {
    default: return vector;                           // 0°
    case 1: return make_float2(-vector.y, vector.x);  // 90°
    case 2: return make_float2(-vector.x, -vector.y); // 180°
    case 3: return make_float2(vector.y, -vector.x);  // 270°
    }
}



} // namespace core::math