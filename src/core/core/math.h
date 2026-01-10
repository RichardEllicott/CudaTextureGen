/*

*/
#pragma once

#include <cstdint>

namespace core::math {

#pragma region CONSTEXPR_HASH

// -----------------------------------------------------------------------------
// splitmix32
// -----------------------------------------------------------------------------
// A tiny, high‑quality 32‑bit mixing function.
// This is NOT a runtime RNG — it's a constexpr hash used to generate
// deterministic, "random‑looking" values at compile time.
//
// Properties:
//   • constexpr‑friendly (no loops, no state)
//   • excellent avalanche behavior
//   • stable across compilers/platforms
//   • ideal for seeds, noise tables, and compile‑time constants
// -----------------------------------------------------------------------------
constexpr uint32_t splitmix32(uint32_t x) {
    x += 0x9e3779b9;
    x = (x ^ (x >> 16)) * 0x85ebca6b;
    x = (x ^ (x >> 13)) * 0xc2b2ae35;
    x ^= (x >> 16);
    return x;
}

// -----------------------------------------------------------------------------
// random32(index)
// -----------------------------------------------------------------------------
// Deterministic constexpr pseudo‑random number generator.
//
// Given an integer index, returns a stable 32‑bit value with good distribution.
// Useful for generating compile‑time tables, seeds, or noise constants without
// maintaining PRNG state.
//
// Notes:
//   • Zero runtime cost
//   • Same output for the same index every build
//   • Uses splitmix32 for high‑quality mixing
// -----------------------------------------------------------------------------
constexpr uint32_t random32(int index) {
    return splitmix32(0xA5A5A5A5u + index * 0x9E3779B9u);
}

// -----------------------------------------------------------------------------
// CONSTEXPR_LINE_SEED
// -----------------------------------------------------------------------------
// Generates a unique, deterministic compile‑time seed based on the call site.
//
// Uses:
//   • __LINE__    → varies by source location
//   • __COUNTER__ → increments per expansion
//
// This macro is ideal when you want a unique seed per use‑site without manually
// managing indices. Each expansion produces a different 32‑bit value.
//
// Example:
//     constexpr uint32_t s0 = CONSTEXPR_LINE_SEED;
//     constexpr uint32_t s1 = CONSTEXPR_LINE_SEED;   // different from s0
//
// Notes:
//   • Must be used at the call site (macro expands in place)
//   • Deterministic across builds unless lines move
// -----------------------------------------------------------------------------

// #define CONSTEXPR_LINE_SEED core::math::splitmix32(uint32_t(__LINE__ * 0x9E3779B9u ^ __COUNTER__))


// this suppresses the warning, i think as random32 takes a normal int
#define CONSTEXPR_LINE_SEED math::random32(__LINE__ * 0x9E3779B9u ^ __COUNTER__)



#pragma endregion

} // namespace core::math