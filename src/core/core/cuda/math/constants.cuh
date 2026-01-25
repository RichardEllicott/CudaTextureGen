/*

maths constants

*/
#pragma once

#include <cstdint> // uint32_t (required for gcc)

#include "core/defines.h"

namespace core::cuda::math {

// ⚠️ adding inline to constexpr can suppress compiler warning
// ⚠️ can't use "std::sqrt(2.0)" in constexpr due to CUDA (so using literals)

constexpr float SQRT2 = 1.4142135623730950488;     // square root of 2
constexpr float INV_SQRT2 = 0.7071067811865475244; // inverse square root of 2
constexpr float PI = 3.14159265358979323846;       // π ratio
constexpr float TAU = PI * 2.0f;

constexpr float DEG_TO_RAD = PI / 180.0f; // multiply to convert degrees to radians
constexpr float RAD_TO_DEG = 180.0f / PI; // multiply to convert radians to degrees

#pragma region HASH

// // magic numbers
// constexpr uint32_t GOLDEN_RATIO_CONST = 0x9E3779B9u; // 32‑bit golden ratio constant (Knuth / SplitMix / xxHash)
// constexpr uint32_t MURMUR3_C1 = 0x85EBCA6Bu;         // MurmurHash3 avalanche constant C1
// constexpr uint32_t MURMUR3_C2 = 0xC2B2AE35u;         // MurmurHash3 avalanche constant C2

// constexpr uint32_t XXH_PRIME32_3 = 1274126177u; // xxHash32 Seed mixing
// constexpr uint32_t XXH_PRIME32_4 = 668265263u;  // xxHash32 Secondary avalanche
// constexpr uint32_t XXH_PRIME32_5 = 374761393u;  // xxHash32 Mix low bits, small inputs

// // numbers used to multiply a hash to floats
// constexpr float INV_U32 = 0x1p-32f; // exactly 2^-32
// constexpr float INV_U31 = 0x1p-31f; // exactly 2^-31
// // constexpr float INV_U30 = 0x1p-30f; // exactly 2^-30

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

} // namespace core::cuda::math
