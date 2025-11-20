/*

🎃 DARRAY TEMPLATE 20251115-1

using new DeviceArray2D ... data is instantly uploaded and downloaded, no local copy

*/
#pragma once

// ================================================================ //
#include "template_macro_undef.h"
#define TEMPLATE_CLASS_NAME Erosion8
#define TEMPLATE_NAMESPACE erosion8

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                                                                   \
    X(bool, debug, false, "track certain information for monitoring")                               \
    X(bool, debug_print, false, "print out information to console")                                 \
    X(int, debug_mod, 1, "frequency to print the debug output")                                     \
    X(size_t, _block, 16, "gpu block size (best at 16)")                                            \
    X(size_t, _width, 512, "map width")                                                             \
    X(size_t, _height, 512, "map height")                                                           \
    X(int, steps, 1024, "simulation steps to run")                                                  \
    X(float, rain_rate, 0.0, "")                                                                    \
    X(bool, rain_random, false, "rain rate is multiplied by a random value from 0 to 1")            \
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
    X(int, erosion_mode, 0, "0 water outflow alone, 1 *slope; 2 *slope soft sat; 3 exp slope")      \
    X(float, slope_exponent, 0.5, "erosion mode 2 only, < 1 soften, > 1 exaggerate")                \
    X(float, deposition_rate, 0.0, "rate sediment becomes height or rock again, deposition_mode 0") \
    X(int, deposition_mode, 0, "0 = basic, 1 =  capacity based")                                    \
    X(float, sediment_capacity, 1.0, "capacity for deposition_mode 1")                              \
    X(float, simple_erosion_rate, 0.0, "simply lower based on the total slope (like Erosion4)")     \
    X(float, slope_threshold, 0.0, "don't count any slope under this value (like Erosion4)")        \
    X(bool, drain_at_min_height, false, "testing drain at min height")

// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_2DS                                           \
    X(float, height_map, "current height, set this map to start the simulation")  \
    X(float, water_map, "current water, optionally set this map at start")        \
    X(float, sediment_map, "current sediment,  optionally set this map at start") \
    X(float, hardness_map, "optional hardness map (not yet used)")                \
    X(float, rain_map, "optional rain map (not yet used)")                        \
    X(float, _height_map_out, "height out")                                       \
    X(float, _water_map_out, "water out")                                         \
    X(float, _sediment_map_out, "sediment out")                                   \
    X(float, _slope_map, "strength of slop")

// (TYPE, DIMENSION, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAYS                      \
    X(float, 8, _flux8, "water flow out to 8 neighbours") \
    X(float, 8, _sediment_flux8, "sediment flow out to 8 neighbours")

// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_3DS \
    X(float, layered_height_map, "testing the idea of a layered heightmap")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_DEBUG_DATA   \
    X(float, total_height, 0.0, "") \
    X(float, total_water, 0.0, "")  \
    X(float, total_sediment, 0.0, "")

// ================================================================ //

#include "core.h"
#include "cuda_types.cuh"

namespace TEMPLATE_NAMESPACE {

// Parameters struct for uploading to GPU
#ifdef TEMPLATE_CLASS_PARAMETERS
struct Parameters {
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
};
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy"); // optional
#endif

// tracking vars for debug
#ifdef TEMPLATE_CLASS_DEBUG_DATA
struct DebugData {
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_DEBUG_DATA
#undef X
};
static_assert(std::is_trivially_copyable<DebugData>::value, "Parameters must remain trivially copyable for CUDA memcpy"); // optional
#endif

class TEMPLATE_CLASS_NAME {

    Parameters pars;                               // local pars
    core::cuda::DeviceStruct<Parameters> dev_pars; // device side pars

    bool pars_synced = false; // record if the pars have been synced since update
    void sync_pars() {
        if (!pars_synced) {
            dev_pars.upload(pars);
            pars_synced = true;
        }
    }

    core::cuda::Stream stream; // will be allocated along with object
    bool device_allocated = false;

    core::cuda::CurandArray2D_2 curand_array_2d;

    core::util::Timer timer;

  public:
    // getter/setters for the pars
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION)   \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) {                 \
        if (pars.NAME != value)                   \
            pars_synced = false;                  \
        pars.NAME = value;                        \
    }
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

    // set default pars back (not needed?)
    void set_defaults_pars() {
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    pars.NAME = DEFAULT_VAL;
        TEMPLATE_CLASS_PARAMETERS
#undef X
        pars_synced = false;
#endif
    }

// DeviceArray's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    core::cuda::DeviceArray<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

// DeviceArray2D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION) \
    core::cuda::DeviceArray2D<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif

// DeviceArray3D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#define X(TYPE, NAME, DESCRIPTION) \
    core::cuda::DeviceArray3D<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#undef X
#endif

    void allocate_device();
    void deallocate_device();

    void process();
};

} // namespace TEMPLATE_NAMESPACE
