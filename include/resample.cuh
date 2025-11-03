/*

resample


sample_mode, bilinear (only one so far)

function_mode 0 = use map_x and map_y (only one so far)





*/
#pragma once

#define RESAMPLE_PARAMETERS        \
    X(size_t, _block, 16)          \
    X(bool, relative_offset, true) \
    X(bool, scale_by_output_size, true)\
    X(int, sample_mode, 0)\
    X(int, function_mode, 0)\

#define RESAMPLE_MAPS \
    X(float, input)   \
    X(float, output)  \
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
    core::cuda::CudaArray2D<TYPE> NAME;
    RESAMPLE_MAPS
#undef X

    // process will take the input and resample to the output using map_x and map_y for offset
    void process();

    void transform_process();
};
} // namespace resample