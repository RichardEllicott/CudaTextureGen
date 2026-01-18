/*


*/
#pragma once

#define D_INLINE __device__ __forceinline__           // device only functions
#define DH_INLINE __device__ __host__ __forceinline__ // device and host functions

// #include "math/constants.cuh"
#include "math/fast.cuh" // intrinsics

namespace core::cuda::hash {

#pragma region CONSTANTS

// magic numbers
constexpr uint32_t GOLDEN_RATIO_CONST = 0x9E3779B9u; // 32‑bit golden ratio constant (Knuth / SplitMix / xxHash)
constexpr uint32_t MURMUR3_C1 = 0x85EBCA6Bu;         // MurmurHash3 avalanche constant C1
constexpr uint32_t MURMUR3_C2 = 0xC2B2AE35u;         // MurmurHash3 avalanche constant C2

constexpr uint32_t XXH_PRIME32_3 = 1274126177u; // xxHash32 Seed mixing
constexpr uint32_t XXH_PRIME32_4 = 668265263u;  // xxHash32 Secondary avalanche
constexpr uint32_t XXH_PRIME32_5 = 374761393u;  // xxHash32 Mix low bits, small inputs

// numbers used to multiply a hash to floats
constexpr float INV_U32 = 0x1p-32f; // exactly 2^-32
constexpr float INV_U31 = 0x1p-31f; // exactly 2^-31
// constexpr float INV_U30 = 0x1p-30f; // exactly 2^-30

#pragma endregion

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
    // return static_cast<float>(hash) * INV_U32; // Scale to [0,1]
    return core::cuda::math::fast::int2float(hash) * INV_U32; // Scale to [0,1]
}

// float from [0,1]
DH_INLINE float hash_float(uint32_t x, uint32_t y, uint32_t z, uint32_t seed) {
    return hash_float(hash_uint(x, y, z, seed));
}

// float from [-1,1]
DH_INLINE float hash_float_signed(int hash) {
    // return static_cast<float>(hash) * INV_U31; // Scale to [-1,1).
    return core::cuda::math::fast::int2float(hash) * INV_U31; // Scale to [0,1]
}

// float from [-1,1] range:
DH_INLINE float hash_float_signed(int x, int y, int z, int seed) {
    return hash_float_signed(hash_int(x, y, z, seed));
}

// take in a hash, extract a bool (set index from 0 to 31)
DH_INLINE bool hash_bool(int hash, int index = 0) {
    return (hash >> index) & 1u;
}

// get 4 random float's from one 32 bit hash, they are not so random though with about 255 possible values
// set byte_index from 0-3
// used only for very low quality random like jitter over frames
DH_INLINE float hash_to_4randf(uint32_t h, int byte_index) {
    uint32_t byte = (h >> (8 * byte_index)) & 0xFFu;
    return (float(byte) / 127.5f) - 1.0f; // // map [0,255] to [-1,1]
}

// --------------------------------------------------------------------------------------------------------------------------------

// SplitMix32 / MurmurHash3 finalizer (high quality)
DH_INLINE uint32_t hash_mix_murmur(uint32_t x) {

    x ^= x >> 16;    // 1) initial avalanche: fold high bits into low bits
    x *= 0x7feb352d; // 2) mix with constant #1 (large odd 32‑bit prime-like)
    x ^= x >> 15;    // 3) second avalanche: spread entropy further
    x *= 0x846ca68b; // 4) mix with constant #2 (another high‑quality mixer)
    x ^= x >> 16;    // 5) final avalanche: ensure full bit diffusion
    return x;        // output: high‑quality 32‑bit mixed value
}

// wang (less quality, cheaper)
// Wang is intentionally minimal: only two multiplies, a few shifts, and XORs
DH_INLINE uint32_t hash_mix_wang(uint32_t x) {
    x = (x ^ 61u) ^ (x >> 16); // 1) initial scramble: inject a fixed odd constant + fold high bits down
    x *= 9u;                   // 2) cheap multiply: small odd constant to spread bits
    x ^= x >> 4;               // 3) secondary avalanche: mix high nibble into low bits
    x *= 0x27d4eb2du;          // 4) main mixing multiply: strong 32‑bit diffusion constant
    x ^= x >> 15;              // 5) final avalanche: destroy remaining linearity
    return x;                  // output: fast, reasonably uniform 32‑bit mix
}

// jenkins (least quality, most cheap)
// No multiplications → extremely fast on GPU
DH_INLINE uint32_t hash_mix_jenkins(uint32_t x) {
    x += (x << 10); // 1) add + shift: quick nonlinear expansion of low bits
    x ^= (x >> 6);  // 2) fold high bits down to increase diffusion
    x += (x << 3);  // 3) second expansion: cheap additive mixing
    x ^= (x >> 11); // 4) another avalanche step to break patterns
    x += (x << 15); // 5) final expansion: push entropy across full 32 bits
    return x;       // output: very cheap, decent-quality mix
}

enum class HashMixType : int {
    Murmur = 0,
    Wang = 1,
    Jenkins = 2
};

DH_INLINE uint32_t hash_mix(uint32_t x, HashMixType type = HashMixType::Murmur) {
    switch (type) {
    case HashMixType::Murmur:
        return hash_mix_murmur(x);
    case HashMixType::Wang:
        return hash_mix_wang(x);
    case HashMixType::Jenkins:
        return hash_mix_jenkins(x);
    }

    std::abort(); // unreachable
}

struct HashRng {
    uint32_t x;

    // output float [-1, 1]
    DH_INLINE float next_signed() {
        x = hash_mix(x);
        return hash_float_signed(x); // [-1,1)
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

} // namespace core::cuda::hash
