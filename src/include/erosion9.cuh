/*

🦑 TEMPLATE_D 20251130-1

2-part, trying constexpr

*/

#pragma once
#include "template_d_base.cuh"

// ================================================================ //
#define TEMPLATE_CLASS_NAME Erosion9
#define TEMPLATE_NAMESPACE erosion9

// Topsoil:
//     Erosion resistance: 0.25
//     Sediment yield: 1.00
//     Permeability: 0.80
//     Erosion threshold: 0.10
//     Color hex: #6B8E23

// Subsoil:
//     Erosion resistance: 0.55
//     Sediment yield: 0.60
//     Permeability: 0.45
//     Erosion threshold: 0.25
//     Color hex: #C2A35B

// Bedrock:
//     Erosion resistance: 0.90
//     Sediment yield: 0.20
//     Permeability: 0.10
//     Erosion threshold: 0.60
//     Color hex: #696969

// using Int2 = std::array<int, 2>;
// #define DEFAULT_INT2 {0, 0}
// #define DEFAULT_INT2 make_int2(0, 0)

using Float3 = std::array<float, 3>;
#define LAYER_NAME_DEFAULT {"Topsoil", "Subsoil", "Bedrock"} // not trivially copyable
#define LAYER_RESISTANCE_DEFAULT {0.25, 0.55, 0.90}          // suggested by ai but changing to
#define LAYER_EROSIVENESS_DEFAULT {1.0, 0.6, 0.1333333}
#define LAYER_YIELD_DEFAULT {1.0, 0.6, 0.2}
#define LAYER_PERMEABILITY_DEFAULT {0.8, 0.25, 0.10}
#define LAYER_THRESHOLD_DEFAULT {0.1, 0.25, 0.6}

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                                                                               \
    X(bool, _debug, false, "track certain information for monitoring")                                          \
    X(bool, debug_print, false, "print out information to console")                                             \
    X(int, debug_mod, 1, "frequency to print the debug output")                                                 \
    X(size_t, _block, 16, "gpu block size (best at 16)")                                                        \
    X(size_t, _width, 512, "map width")                                                                         \
    X(size_t, _height, 512, "map height")                                                                       \
    X(float, _calculation_time, 0.0, "saving the calculation time")                                             \
    X(int, steps, 1024, "simulation steps to run")                                                              \
    X(float, rain_rate, 0.0, "")                                                                                \
    X(bool, rain_random, false, "rain rate is multiplied by a random value from 0 to 1")                        \
    X(bool, wrap, true, "wrap the errosion from one side to the other (making result tileable)")                \
    X(float, max_water_outflow, 1000000.0, "max outflow from a cell per a turn")                                \
    X(float, diffusion_rate, 0.0, "try to diffuse water away from the slops, 0.0 is off")                       \
    X(bool, correct_diagonal_distance, true, "normally true, makes sure diagonals are ~1.4 away")               \
    X(float, slope_jitter, 0.0, "added jitter to the calculate slope values")                                   \
    X(float, outflow_carve, 0.0, "reduce height based on outflow (no sediment)")                                \
    X(float, min_height, -1000000.0, "minimum height the terrain can erode down to")                            \
    X(float, max_height, 1000000.0, "maximum height the terrain can erode down to")                             \
    X(float, evaporation_rate, 0.0, "speed at which water disappears")                                          \
    X(float, erosion_rate, 0.0, "rate at which height becomes sediment based on water outflow")                 \
    X(int, erosion_mode, 0, "0 water outflow alone, 1 *slope; 2 *slope soft sat; 3 exp slope")                  \
    X(float, slope_exponent, 0.5, "erosion mode 2 only, < 1 soften, > 1 exaggerate")                            \
    X(float, deposition_rate, 0.0, "rate sediment becomes height or rock again, deposition_mode 0")             \
    X(int, deposition_mode, 0, "0 = basic, 1 =  capacity based")                                                \
    X(float, sediment_capacity, 0.0, "⚠️ WARNING I CHANGED FROM 1.0 to 0.0 !!!! capacity for deposition_mode 1") \
    X(float, simple_erosion_rate, 0.0, "simply lower based on the total slope (like Erosion4)")                 \
    X(float, slope_threshold, 0.0, "don't count any slope under this value (like Erosion4)")                    \
    X(bool, drain_at_min_height, false, "testing drain at min height")                                          \
    X(float, drain_rate, 1000000.0, "rate of drain")                                                            \
    X(int, mode, 0, "🚧 different modes for serious refactors")                                                 \
    X(size_t, _layers, 3, "❌ total layers")                                                                    \
    X(Float3, layers_erosiveness, LAYER_EROSIVENESS_DEFAULT, "❌ multiply by erosion_rate")                     \
    X(Float3, layers_yield, LAYER_YIELD_DEFAULT, "❌ sediment to release")                                      \
    X(Float3, layers_permeability, LAYER_PERMEABILITY_DEFAULT, "❌ not sure?")                                  \
    X(Float3, layers_threshold, LAYER_THRESHOLD_DEFAULT, "❌ not sure?")                                        \
    X(float, scale, 1.0, "🐙 real world width of a pixel")                                                      \
    X(float, gravity, -9.8, "❌ gravity with regard to positive being upwards")                                 \
    X(float, flow_rate, 1.0, "🐙 flow rate for new model")                                                      \
    X(float, sediment_yield, 1.0, "🐙 amount of sediment generated")                                            \
    X(float, positive_slope_gradient_cap, 1000000.0, "🐙 amount of sediment generated")

#define TEMPLATE_DEBUG_OUTPUTS                                    \
    X(float, _debug_rain_total, 0.0, "tracking total rain")       \
    X(float, _debug_drain_total, 0.0, "tracking total drain")     \
    X(float, _debug_erosion_total, 0.0, "tracking total erosion") \
    X(float, _debug_evaporation_total, 0.0, "tracking total erosion")

// X(int2, test_int2, DEFAULT_INT2, "test a Int2")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_NS                                                               \
    X(float, 2, height_map, "current height, set this map to start the simulation")                  \
    X(float, 2, _height_map_out, "height out")                                                       \
    X(float, 2, water_map, "current water, optionally set this map at start")                        \
    X(float, 2, _water_map_out, "water out")                                                         \
    X(float, 2, sediment_map, "current sediment,  optionally set this map at start")                 \
    X(float, 2, _sediment_map_out, "sediment out")                                                   \
    X(float, 2, rain_map, "optional rain map, multiply by this")                                     \
    X(float, 2, _slope_map, "strength of slope")                                                     \
    X(float, 1, _flux8, "⚠️ 8 water flow out to 8 neighbours")                                        \
    X(float, 1, _sediment_flux8, "⚠️ sediment flow out to 8 neighbours")                              \
    X(float, 3, layer_map, "layered version of height_map, should be filled with 3 layers from RGB") \
    X(float, 3, _layer_map_out, "layer out")                                                         \
    X(float, 2, _surface_map, "calculating the surface height")                                      \
    X(float, 2, hardness_map, "optional hardness map")

// X(float, 2, hardness_map, "optional hardness map (not yet used)")

// ================================================================ //

namespace TEMPLATE_NAMESPACE {

// Parameters struct for uploading to GPU
struct Parameters {
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif
};
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy"); // optional

// ⚠️ Array struct for uploading to GPU
struct ArrayPtrs {
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    TYPE *NAME = nullptr;
    TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif
};
static_assert(std::is_trivially_copyable<ArrayPtrs>::value, "ArrayPtrs must remain trivially copyable for CUDA memcpy"); // optional

// Parameters struct for uploading to GPU
struct DebugOutputs {
#ifdef TEMPLATE_DEBUG_OUTPUTS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_DEBUG_OUTPUTS
#undef X
#endif
};
static_assert(std::is_trivially_copyable<DebugOutputs>::value, "DebugOutputs must remain trivially copyable for CUDA memcpy"); // optional

class TEMPLATE_CLASS_NAME : public template_d::TemplateD<Parameters> {

  protected:
    core::cuda::DeviceSyncedStruct<DebugOutputs> debug_outputs; // device side pars (new synced wrapper keeps a local copy)
    core::cuda::DeviceStruct<ArrayPtrs> dev_array_ptrs;         // device side pars

  public:
    // getter/setters for the pars
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION)   \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { set_par(pars.NAME, value); }
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

    // getter/setters for the debug output's
#ifdef TEMPLATE_DEBUG_OUTPUTS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION)                   \
    TYPE get_##NAME() const { return debug_outputs.host().NAME; } \
    void set_##NAME(TYPE value) { debug_outputs.host().NAME = value; }
    TEMPLATE_DEBUG_OUTPUTS
#undef X
#endif

// DeviceArray's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    core::cuda::DeviceArray<TYPE, DIMENSIONS> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif

    // lazy function to return array pointers
    std::vector<core::cuda::DeviceArrayBase *> get_device_array_n_ptrs() override {
        if (_device_array_n_ptrs.empty()) {
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    _device_array_n_ptrs.push_back(&NAME);
            TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif
        }
        return _device_array_n_ptrs;
    }

    core::cuda::CurandArray2D curand_array_2d;

    // get pointers to the arrays
    ArrayPtrs get_array_ptrs() {
        return {
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    NAME.dev_ptr(),
            TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif
        };
    }

    // // (TYPE, DIMENSIONS, NAME, DESCRIPTION)
    // struct ArrayMeta {
    //     const char *type;        // e.g. "float", "int", "double"
    //     const int dimensions;    // e.g. "1D", "2x2", "NxM"
    //     const char *name;        // identifier
    //     const char *description; // human-readable docstring
    // };

    void process00();
    void process01();

    TEMPLATE_CLASS_NAME() {
        initialize();
        debug_outputs.set_stream(stream.get());
        dev_array_ptrs.set_stream(stream.get());
    }

    void allocate_device() override;

    void allocate_device00();
    void allocate_device01();

    void process() override;
    void debug_update();
};

} // namespace TEMPLATE_NAMESPACE
