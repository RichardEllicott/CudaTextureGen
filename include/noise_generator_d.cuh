/*

looking for 3D noise variations


type      -nose type
seed      -seed
period    -like frequency, set as a whole number for seamless noise
x         -start position x
y         -start position y
z         -start position z (for 3D noise, useful for animation)

*/
#pragma once

#include "noise_util.cuh"
#include <iostream>

// #define NOISE_GENERATOR_D_HASH_MODE 0 // bitwise, trig based // 🚧 UNUSED

#define NOISE_GENERATOR_D_PARAMETERS \
    X(int, type, 0)                  \
    X(int, seed, 0)                  \
    X(float, period, 8.0f)           \
    X(float, scale, 1.0f)            \
    X(float, x, 0.0f)                \
    X(float, y, 0.0f)                \
    X(float, z, 0.0f)                \
    X(float, warp_amp, 4.0f)         \
    X(float, warp_scale, 1.0f)

// // 🚧 xmacro extension idea
// #define NOISE_GENERATOR_D_ADVANCED_PARAMETERS \
//     X(float, gain, 0.5f)                      \
//     X(float, lacunarity, 2.0f)                \
//     X(int, octaves, 4)

// // 🚧 xmacro extension idea
// #define NOISE_GENERATOR_D_ALL_PARAMETERS \
//     NOISE_GENERATOR_D_PARAMETERS         \
//     NOISE_GENERATOR_D_ADVANCED_PARAMETERS

#define NOISE_GENERATOR_D_TYPES \
    X(Value2D)                  \
    X(Gradient2D)               \
    X(WarpedValue2D)            \
    X(Value3D)                  \
    X(Gradient3D)               \
    X(Hash2D)                   \
    X(Hash3D)

namespace noise_generator_d {

struct Parameters {
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE NAME;
    NOISE_GENERATOR_D_PARAMETERS
#undef X
};

class NoiseGeneratorD {

  private:
    Parameters pars;

  public:
    NoiseGeneratorD() {
        // set default values
#define X(TYPE, NAME, DEFAULT_VAL) \
    pars.NAME = DEFAULT_VAL;
        NOISE_GENERATOR_D_PARAMETERS
#undef X
    }
    // make get/sets
#define X(TYPE, NAME, DEFAULT_VAL)                \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
        NOISE_GENERATOR_D_PARAMETERS
#undef X

    // make enumerators
    enum class Type {

#define X(NAME) \
    NAME,
        NOISE_GENERATOR_D_TYPES
#undef X
    };

    void fill(float *d_out, const unsigned width, const unsigned height);
};

} // namespace noise_generator_d