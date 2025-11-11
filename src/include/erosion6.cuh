/*

Erosion5

Water and Sediment Flow based erosion with seperate flux and apply pass



other ideas..

Diffusion Blur (Heat Equation) ... similar to basic bank erosion




🧜‍♀️ TEMPLATE VERSION 20251027-3

*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME Erosion6
#define TEMPLATE_NAMESPACE erosion6

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                                                                   \
    X(bool, debug, false, "track certain information for monitoring")                               \
    X(int, debug_mod, 1, "frequency to print the debug output")                                     \
    X(size_t, _block, 16, "gpu block size (best at 16)")                                            \
    X(size_t, _width, 512, "map width")                                                             \
    X(size_t, _height, 512, "map height")                                                           \
    X(int, steps, 1024, "simulation steps to run")                                                  \
    X(float, rain_rate, 0.0, "")                                                                    \
    X(bool, wrap, true, "wrap the errosion from one side to the other (making result tileable)")    \
    X(float, max_water_outflow, 1000000.0, "max outflow from a cell per a turn")                    \
    X(float, diffusion_rate, 0.0, "try to diffuse water away from the slops, 0.0 is off")           \
    X(bool, correct_diagonal_distance, true, "normally true, makes sure diagonals are ~1.4 away")   \
    X(float, slope_jitter, 0.0, "added jitter to the calculate slope values")                       \
    X(float, outflow_carve, 0.0, "reduce height based on outflow (no sediment)")                    \
    X(float, min_height, -1000000.0, "minimum height the terrain can erode down to")                \
    X(float, max_height, 1000000.0, "maximum height the terrain can erode down to")                 \
    X(float, evaporation_rate, 0.0, "speed at which water disappears")                              \
    X(float, erosion_rate, 0.0, "rate at which height becomes sediment based on water outflow")     \
    X(int, erosion_mode, 0, "0 *slope; 1 *slope soft sat; 2 exp slope")                             \
    X(float, slope_exponent, 0.5, "erosion mode 2 only, < 1 soften, > 1 exaggerate")                \
    X(float, deposition_rate, 0.0, "rate sediment becomes height or rock again, deposition_mode 0") \
    X(int, deposition_mode, 0, "0 = basic, 1 =  capacity based")                                    \
    X(float, sediment_capacity, 1.0, "capacity for deposition_mode 1")                              \
    X(float, simple_erosion_rate, 0.0, "simply lower based on the total slope (like Erosion4)")     \
    X(float, slope_threshold, 0.0, "don't count any slope under this value (like Erosion4)")

// X(float, capacity, 0.1)          \
    // X(float, erode, 0.1)             \
    // X(float, deposit, 0.1)           \
    // X(float, evaporation_rate, 0.1)  \
    // X(bool, debug_hash_cell_order, false)\

// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_MAPS                                        \
    X(float, height_map, "starting height")                        \
    X(float, water_map, "starting water")                          \
    X(float, sediment_map, "starting sediment")                    \
    X(float, hardness_map, "optional hardness map (not yet used)") \
    X(float, rain_map, "optional rain map (not yet used)")

// (TYPE, DIMENSION, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAYS                                 \
    X(float, 1, height_map_out, "second height buffer")              \
    X(float, 1, water_map_out, "second water buffer")                \
    X(float, 1, sediment_map_out, "second sediment buffer")          \
    X(float, 8, flux8, "water flow out to 8 neighbours")             \
    X(float, 8, sediment_flux8, "sediment flow out to 8 neighbours") \
    X(float, 1, slope_map, "strength of slope")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_DEBUG_DATA   \
    X(float, total_height, 0.0, "") \
    X(float, total_water, 0.0, "")  \
    X(float, total_sediment, 0.0, "")

// ════════════════════════════════════════════════ //

#include "cuda_types.cuh"

namespace TEMPLATE_NAMESPACE {

// pars structure
struct Parameters {
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
};
// OPTIONAL Compile‑time safety check
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy");

// tracking vars for debug
struct DebugData {
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_DEBUG_DATA
#undef X
};
// OPTIONAL Compile‑time safety check
static_assert(std::is_trivially_copyable<DebugData>::value, "Parameters must remain trivially copyable for CUDA memcpy");

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
#define X(TYPE, NAME, DESCRIPTION) \
    core::cuda::CudaArray2D<TYPE> NAME;
    TEMPLATE_CLASS_MAPS
#undef X

    // private device arrays
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
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
