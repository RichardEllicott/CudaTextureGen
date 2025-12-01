/*

🎃 DARRAY TEMPLATE 20251115-1

refactoring towards layers... didn't make really any changes so keep frozen?

*/
#pragma once

#include <array>
#include <string>

// #include <nanobind/nanobind.h>

// ================================================================ //
#include "template_macro_undef.h"
#define TEMPLATE_CLASS_NAME Erosion8
#define TEMPLATE_NAMESPACE erosion8

//
//
//
// 🚧 🚧 🚧 🚧

using Float2 = std::array<float, 2>;
using Float3 = std::array<float, 3>;
using Float4 = std::array<float, 4>;
using Float5 = std::array<float, 5>;
using Float6 = std::array<float, 6>;
using Float7 = std::array<float, 7>;
using Float8 = std::array<float, 8>;

using String3 = std::array<std::string, 3>; // not trivially copyable

// #define FLOAT4_DEFAULT {1.0f, 0.0f, 0.0f, 1.0f}

#define LAYER_NAME_DEFAULT {"Topsoil", "Subsoil", "Bedrock"} // not trivially copyable
#define LAYER_RESISTANCE_DEFAULT {0.25, 0.55, 0.90}          // suggested by ai but changing to
#define LAYER_EROSIVENESS_DEFAULT {1.0, 0.6, 0.1333333}
#define LAYER_YIELD_DEFAULT {1.0, 0.6, 0.2}
#define LAYER_PERMEABILITY_DEFAULT {0.8, 0.25, 0.10}
#define LAYER_THRESHOLD_DEFAULT {0.1, 0.25, 0.6}

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_EXTRA_PARAMETERS                                                    \
    X(String3, layers_name, LAYER_NAME_DEFAULT, "layer names, but not trivially copyable") \
    X(Float3, layers_resistance, LAYER_RESISTANCE_DEFAULT, "")                             \
    X(Float3, layers_erosiveness, LAYER_EROSIVENESS_DEFAULT, "multiply by erosion_rate")   \
    X(Float3, layers_yield, LAYER_YIELD_DEFAULT, "sediment to release")                    \
    X(Float3, layers_permeability, LAYER_PERMEABILITY_DEFAULT, "not sure?")                \
    X(Float3, layers_threshold, LAYER_THRESHOLD_DEFAULT, "not sure?")
#undef TEMPLATE_CLASS_EXTRA_PARAMETERS

/*

0.75 / 0.75 = 1.0
0.45 / 0.75 = 0.6
0.10 / 0.75 = 0.1333...




Topsoil:
    Erosion resistance: 0.25
    Sediment yield: 1.00
    Permeability: 0.80
    Erosion threshold: 0.10
    Color hex: #6B8E23

Subsoil:
    Erosion resistance: 0.55
    Sediment yield: 0.60
    Permeability: 0.45
    Erosion threshold: 0.25
    Color hex: #C2A35B

Bedrock:
    Erosion resistance: 0.90
    Sediment yield: 0.20
    Permeability: 0.10
    Erosion threshold: 0.60
    Color hex: #696969

Resistance scales erosion rate: effective_rate = base_rate × (1 − resistance).
Sediment yield scales how much eroded mass becomes transportable sediment.
Permeability controls retained water fraction influencing erosion strength next step.
Erosion threshold is the minimum shear/flow needed before erosion starts for that layer.
Colors help visualize progression; tweak to taste.
*/

//
//
//
//

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
    X(bool, drain_at_min_height, false, "testing drain at min height")                              \
    X(int, mode, 0, "🚧 different modes for serious refactors")                                     \
    X(size_t, _layers, 3, "🚧 total layers")                                                        \
    X(Float3, layers_erosiveness, LAYER_EROSIVENESS_DEFAULT, "multiply by erosion_rate")            \
    X(Float3, layers_yield, LAYER_YIELD_DEFAULT, "sediment to release")                             \
    X(Float3, layers_permeability, LAYER_PERMEABILITY_DEFAULT, "not sure?")                         \
    X(Float3, layers_threshold, LAYER_THRESHOLD_DEFAULT, "not sure?")

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
    X(float, height_map3, "testing the idea of a layered heightmap")

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

    core::cuda::CurandArray2D curand_array_2d;

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
    core::cuda::DeviceArray1D<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

// // DeviceArray2D's
// #ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
// #define X(TYPE, NAME, DESCRIPTION) \
//     core::cuda::DeviceArray2D<TYPE> NAME;
//     TEMPLATE_CLASS_DEVICE_ARRAY_2DS
// #undef X
// #endif

// DeviceArrayN2D's
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
