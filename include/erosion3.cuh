/*

🧜‍♀️ TEMPLATE VERSION 20251027-3

this one uses more clever types and classes, needing less horrible macros

*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME Erosion3
#define TEMPLATE_NAMESPACE erosion_3

// #define TEMPLATE_CLASS_PARAMETERS    \
//     X(size_t, width, 256)            \
//     X(size_t, height, 256)           \
//     X(size_t, _block, 16)            \
//     X(float, min_height, 0.0)        \
//     X(float, max_height, 1.0)        \
//     X(float, jitter, 0.0)            \
//     X(float, rain_rate, 0.0)         \
//     X(float, evaporation_rate, 0.01) \
//     X(float, erosion_rate, 0.01)     \
//     X(float, deposition_rate, 0.01)  \
//     X(float, slope_threshold, 0.01)  \
//     X(float, flow_factor, 0.1)       \
//     X(int, steps, 1024)              \
//     X(int, block_size, 16)           \
//     X(int, mode, 0)                  \
//     X(bool, wrap, true)              \
//     X(float, w_max, 1.0)             \
//     X(float, k_capacity, 0.1)        \
//     X(float, k_erode, 0.1)           \
//     X(float, k_deposit, 0.1)         \
//     X(float, evap, 0.1)

#define TEMPLATE_CLASS_PARAMETERS \
    X(size_t, width, 256)         \
    X(size_t, height, 256)        \
    X(int, steps, 1024)           \
    X(size_t, _block, 16)         \
    X(float, rain_rate, 0.01f)      \
    X(bool, wrap, true)           \
    X(float, w_max, 1.0f)          \
    X(float, capacity, 0.1f)     \
    X(float, erode, 0.1f)        \
    X(float, deposit, 0.1f)      \
    X(float, evap, 0.1f)

// pars.rain_rate;
// pars.evap_rate;
// pars.w_max;
// pars.k_capacity;
// pars.k_erode;
// pars.k_deposit;
// pars.wrap;
// pars.epsilon;

#define TEMPLATE_CLASS_MAPS \
    X(float, height_map)    \
    X(float, water_map)     \
    X(float, sediment_map)  \
    X(float, dh_out)        \
    X(float, ds_out)        \
    X(float, dw_out)        \
    // X(FluxCell, flux8)

// #define TEMPLATE_CLASS_TYPES \
//     X(APPLE)                 \
//     X(ORANGE)                \
//     X(POTATO)

// ════════════════════════════════════════════════ //

#include "cuda_types.cuh"

namespace TEMPLATE_NAMESPACE {

// struct FluxCell {
//     float f[8];
// };

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

    // make enumerators
#ifdef TEMPLATE_CLASS_TYPES
    enum class Type {
#define X(NAME) \
    NAME,
        TEMPLATE_CLASS_TYPES
#undef X
    };
#endif

    void process();
};

} // namespace TEMPLATE_NAMESPACE
