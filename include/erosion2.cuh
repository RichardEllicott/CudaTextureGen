/*

Erosion2 based on:

🧜‍♀️ TEMPLATE VERSION 20251027-3

*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME Erosion2
#define TEMPLATE_NAMESPACE erosion_2

#define TEMPLATE_CLASS_PARAMETERS    \
    X(size_t, width, 256)            \
    X(size_t, height, 256)           \
    X(size_t, _block, 16)            \
    X(float, min_height, 0.0)        \
    X(float, max_height, 1.0)        \
    X(float, jitter, 0.0)            \
    X(float, rain_rate, 0.0)         \
    X(float, evaporation_rate, 0.01) \
    X(float, erosion_rate, 0.01)     \
    X(float, deposition_rate, 0.01)  \
    X(float, slope_threshold, 0.01)  \
    X(float, flow_factor, 0.1)       \
    X(int, steps, 1024)              \
    X(int, block_size, 16)           \
    X(int, mode, 0)                  \
    X(bool, wrap, true)              \
    X(float, w_max, 1.0)             \
    X(float, k_capacity, 0.1)        \
    X(float, k_erode, 0.1)           \
    X(float, k_deposit, 0.1)         \
    X(float, evap, 0.1)

#define TEMPLATE_CLASS_MAPS \
    X(float, height_map)    \
    X(float, sediment_map)  \
    X(float, water_map)

// ════════════════════════════════════════════════ //

#include "cuda_types.cuh"

namespace TEMPLATE_NAMESPACE {

struct Parameters {
    // declare pars on structures
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
};

class TEMPLATE_CLASS_NAME {

  private:
    Parameters pars;

  public:
    // make getter/setters for the pars
#define X(TYPE, NAME, DEFAULT_VAL)                \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
    TEMPLATE_CLASS_PARAMETERS
#undef X

// make maps
#define X(TYPE, NAME) \
    core::CudaArray2D<TYPE> NAME;
    TEMPLATE_CLASS_MAPS
#undef X

    void process();
};

} // namespace TEMPLATE_NAMESPACE
