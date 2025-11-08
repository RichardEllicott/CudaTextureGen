/*

🧜‍♀️ TEMPLATE VERSION 20251027-3



THE ERROSION WE HAD WORKING... SEEMS TO BREAL WITH THE CORRECT PARS!!





*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME Erosion5
#define TEMPLATE_NAMESPACE erosion5

#define TEMPLATE_CLASS_PARAMETERS                                                         \
    X(size_t, _block, 16, "")                                                             \
    X(size_t, width, 256, "")                                                             \
    X(size_t, height, 256, "")                                                            \
    X(int, steps, 1024, "")                                                               \
    X(float, rain_rate, 0.0, "")                                                          \
    X(bool, wrap, true, "")                                                               \
    X(float, max_water_outflow, 1000000.0, "max outflow from a cell per a turn")          \
    X(float, diffusion_rate, 0.0, "try to diffuse water away from the slops, 0.0 is off") \
    X(bool, correct_diagonal_distance, true, "")                                          \
    X(float, slope_jitter, 0.0, "")                                                       \
    X(float, outflow_erode, 0.0, "reduce height based on outflow")                        \
    X(float, inflow_erode, 0.0, "reduce height based on inflow")                          \
    X(float, min_height, -1000000.0, "minimum height the terrain can erode down to")      \
    X(float, max_height, 1000000.0, "maximum height the terrain can erode down to")       \
    X(float, evaporation_rate, 0.0, "")

// X(float, capacity, 0.1)          \
    // X(float, erode, 0.1)             \
    // X(float, deposit, 0.1)           \
    // X(float, evaporation_rate, 0.1)  \
    // X(bool, debug_hash_cell_order, false)\

#define TEMPLATE_CLASS_MAPS \
    X(float, height_map)    \
    X(float, water_map)     \
    X(float, sediment_map)

// private device arrays X(TYPE, NAME, Z_SIZE)
#define TEMPLATE_CLASS_DEVICE_ARRAYS \
    X(float, height_map_out, 1)      \
    X(float, water_map_out, 1)       \
    X(float, sediment_map_out, 1)    \
    X(float, dh_out, 1)              \
    X(float, ds_out, 1)              \
    X(float, dw_out, 1)              \
    X(float, flux8, 8)

// ════════════════════════════════════════════════ //

#include "cuda_types.cuh"

namespace TEMPLATE_NAMESPACE {

// struct FluxCell {
//     float f[8];
// };

struct Parameters {
    // declare pars on structures
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
};

class TEMPLATE_CLASS_NAME {

  private:
    Parameters pars;
    bool device_allocated = false;

    size_t _count = 0; // count of passes

  public:
    // make getter/setters for the pars
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION)   \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
    TEMPLATE_CLASS_PARAMETERS
#undef X

    // make maps
#define X(TYPE, NAME) \
    core::cuda::CudaArray2D<TYPE> NAME;
    TEMPLATE_CLASS_MAPS
#undef X

    // private device arrays
#define X(TYPE, NAME, Z_SIZE) \
    core::cuda::DeviceArray<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X

    void allocate_device(); // and upload
    // void upload_device();
    void deallocate_device(); // and download
    // void download_device();

    void process();

    TEMPLATE_CLASS_NAME();
    ~TEMPLATE_CLASS_NAME();
};

} // namespace TEMPLATE_NAMESPACE
