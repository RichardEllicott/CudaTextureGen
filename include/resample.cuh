/*



*/
#pragma once

#define RESAMPLE_PARAMETERS \
    X(float, test1, 0.0f)   \
    X(float, test2, 0.0f)   \
    X(float, test3, 0.0f)

#define RESAMPLE_MAPS \
    X(float, input)   \
    X(float, output)   \
    X(float, map_x)   \
    X(float, map_y)

// #include "core.h"
// #include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#include "cuda_types.cuh"

namespace resample {

struct Parameters {
    // declare pars on structures
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE NAME = DEFAULT_VAL;
    RESAMPLE_PARAMETERS
#undef X
};

class Resample {
  private:
    Parameters pars;

  public:
    // make getter/setters for the pars
#define X(TYPE, NAME, DEFAULT_VAL)                \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
    RESAMPLE_PARAMETERS
#undef X

// make maps
#define X(TYPE, NAME) \
    core::CudaArray2D<TYPE> NAME;
    RESAMPLE_MAPS
#undef X

    void process_maps(
        const int width, const int height,
        const float *host_in, float *host_out,
        const float *map_x, const float *map_y);

    void process();
};
} // namespace resample