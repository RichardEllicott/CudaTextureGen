/*

host side maths

*/
#pragma once

#include <cstdint> // uint32_t

#include "cuda_compat.h"

namespace core::hash {

// ================================================================================================================================

// Full-period 32-bit reversible permutation
// Visits all 2^32 states exactly once
inline uint32_t permute32(uint32_t x) {
    x ^= x >> 17; // 1
    // --------------------------------
    x *= 0xed5ad4bbu; // 2
    // --------------------------------
    x ^= x >> 11; // 3
    // --------------------------------
    x *= 0xac4c1b51u; // 4
    // --------------------------------
    x ^= x >> 15; // 5
    // --------------------------------
    x *= 0x31848babu; // 6
    // --------------------------------
    x ^= x >> 14; // 7
    // --------------------------------
    return x;
}

// // Inverse of permute32
// inline uint32_t permute32_inverse(uint32_t x) {
//     x ^= x >> 14;
//     x ^= x >> 28; // 7
//     // --------------------------------
//     x *= 0x32b21703u; // 6 modular inverse of 0x31848bab
//     // --------------------------------
//     x ^= x >> 15;
//     x ^= x >> 30; // 5
//     // --------------------------------
//     x *= 0x7a1ba0d9u; // 4 modular inverse of 0xac4c1b51
//     // --------------------------------
//     x ^= x >> 11;
//     x ^= x >> 22; // 3
//     // --------------------------------
//     x *= 0xa0d1b2c3u; // 2 modular inverse of 0xed5ad4bb
//     // --------------------------------
//     x ^= x >> 17;
//     x ^= x >> 34; // 1 wraps to 2 bits in 32-bit domain
//                   // --------------------------------
//     return x;
// }

inline uint32_t permute32_inverse(uint32_t x) {
    x ^= x >> 14;
    x ^= x >> 28; // 7
    // --------------------------------
    x *= 0x32b21703u; // inverse of 0x31848bab
    // --------------------------------
    x ^= x >> 15;
    x ^= x >> 30; // 5
    // --------------------------------
    x *= 0x7a1ba0d9u; // inverse of 0xac4c1b51
    // --------------------------------
    x ^= x >> 11;
    x ^= x >> 22; // 3
    // --------------------------------
    x *= 0xa0d1b2c3u; // inverse of 0xed5ad4bb
    // --------------------------------
    x ^= x >> 17;
    x ^= x >> (34 & 31); // 1 → 2-bit shift
    // --------------------------------
    return x;
}

/*
reversable:

1. XOR
2. ADD / SUBTRACT (mod 2ⁿ)
3. MULTIPLY by an odd constant
    -Reversible because odd numbers have a modular inverse modulo 2^n
4. ROTATE (ROL / ROR)

Conditionally reversible operations
5. XOR‑shift
    -Reversible if you apply the correct sequence of inverse shifts
6. NEGATION

7. REVERSING BITS


multiply requires some computation for reverse

MULTIPLY by an odd constant

is reversible because there exists a number k_inv such that:

x * k * k_inv ≡ x (mod 2^n)



*/

// --------------------------------------------------------------------------------------------------------------------------------

} // namespace core::hash