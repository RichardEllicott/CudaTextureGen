/*

👻 TEMPLATE_E 20251213

*/

#pragma once
#include "template_e_base.cuh"
#include <stdexcept>
#include <unordered_set>

// ================================================================ //
#define TEMPLATE_CLASS_NAME Erosion10
#define TEMPLATE_NAMESPACE erosion10

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
using Float2 = std::array<float, 2>;

#define LAYER_NAME_DEFAULT {"Topsoil", "Subsoil", "Bedrock"} // not trivially copyable
#define LAYER_RESISTANCE_DEFAULT {0.25, 0.55, 0.90}          // suggested by ai but changing to
#define LAYER_EROSIVENESS_DEFAULT {1.0, 0.6, 0.1333333}
#define LAYER_YIELD_DEFAULT {1.0, 0.6, 0.2}
#define LAYER_PERMEABILITY_DEFAULT {0.8, 0.25, 0.10}
#define LAYER_THRESHOLD_DEFAULT {0.1, 0.25, 0.6}

// // (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
// #define TEMPLATE_CLASS_PARAMETERS                                                                   \
//     X(bool, _debug, false, "track certain information for monitoring")                              \
//     X(bool, debug_print, false, "print out information to console")                                 \
//     X(int, debug_mod, 1, "frequency to print the debug output")                                     \
//     X(size_t, _block, 16, "gpu block size (best at 16)")                                            \
//     X(size_t, _width, 512, "map width")                                                             \
//     X(size_t, _height, 512, "map height")                                                           \
//     X(float, _calculation_time, 0.0, "saving the calculation time")                                 \
//     X(int, steps, 1024, "simulation steps to run")                                                  \
//     X(float, rain_rate, 0.0, "")                                                                    \
//     X(bool, rain_random, false, "rain rate is multiplied by a random value from 0 to 1")            \
//     X(bool, wrap, true, "wrap the errosion from one side to the other (making result tileable)")    \
//     X(float, max_water_outflow, 1000000.0, "max outflow from a cell per a turn")                    \
//     X(float, diffusion_rate, 0.0, "try to diffuse water away from the slops, 0.0 is off")           \
//     X(bool, correct_diagonal_distance, true, "normally true, makes sure diagonals are ~1.4 away")   \
//     X(float, slope_jitter, 0.0, "added jitter to the calculate slope values")                       \
//     X(float, outflow_carve, 0.0, "reduce height based on outflow (no sediment)")                    \
//     X(float, min_height, -1000000.0, "minimum height the terrain can erode down to")                \
//     X(float, max_height, 1000000.0, "maximum height the terrain can erode down to")                 \
//     X(float, evaporation_rate, 0.0, "speed at which water disappears")                              \
//     X(float, erosion_rate, 0.0, "rate at which height becomes sediment based on water outflow")     \
//     X(int, erosion_mode, 0, "0 water outflow alone, 1 *slope; 2 *slope soft sat; 3 exp slope")      \
//     X(float, slope_exponent, 0.5, "erosion mode 2 only, < 1 soften, > 1 exaggerate")                \
//     X(float, deposition_rate, 0.0, "rate sediment becomes height or rock again, deposition_mode 0") \
//     X(int, deposition_mode, 0, "0 = basic, 1 =  capacity based")                                    \
//     X(float, sediment_capacity, 0.0, "sediment capacity of the water")                              \
//     X(float, simple_erosion_rate, 0.0, "simply lower based on the total slope (like Erosion4)")     \
//     X(float, slope_threshold, 0.0, "don't count any slope under this value (like Erosion4)")        \
//     X(float, drain_rate, 0.0, "rate of water drain when reaching minimum height")                   \
//     X(float, sediment_drain_rate, 0.0, "rate of sediment drain when reaching minimum height")       \
//     X(int, mode, 0, "🚧 different modes for serious refactors")                                     \
//     X(size_t, _layers, 3, "❌ total layers")                                                        \
//     X(Float3, layers_erosiveness, LAYER_EROSIVENESS_DEFAULT, "❌ multiply by erosion_rate")         \
//     X(Float3, layers_yield, LAYER_YIELD_DEFAULT, "❌ sediment to release")                          \
//     X(Float3, layers_permeability, LAYER_PERMEABILITY_DEFAULT, "❌ not sure?")                      \
//     X(Float3, layers_threshold, LAYER_THRESHOLD_DEFAULT, "❌ not sure?")                            \
//     X(float, scale, 1.0, "🐙 real world width of a pixel")                                          \
//     X(float, gravity, -9.8, "❌ gravity with regard to positive being upwards")                     \
//     X(float, flow_rate, 1.0, "🐙 flow rate for new model")                                          \
//     X(float, sediment_yield, 0.0, "🐙 amount of sediment generated")                                \
//     X(float, positive_slope_gradient_cap, 1000000.0, "🐙 amount of sediment generated")             \
//     X(float, deposition_threshold, 0.0, "🐙 amount of sediment generated")                          \
//     X(int, slope_jitter_mode, 0, "0 is based on step, 1 is frozen")                                 \
//     X(int, manning_mode, 0, "for second erorsion model only, trying different manning calculations")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                                                                                   \
    X(bool, _debug, false, "track certain information for monitoring")                                              \
    X(bool, debug_print, false, "print out information to console")                                                 \
    X(int, debug_mod, 1, "frequency to print the debug output")                                                     \
    X(size_t, _block, 16, "gpu block size (best at 16)")                                                            \
    X(size_t, _width, 512, "map width")                                                                             \
    X(size_t, _height, 512, "map height")                                                                           \
    X(size_t, _layers, 0, "layers for layer mode")                                                                  \
    X(int, mode, 0, "❌ [0]: Default, [1]: Test Wind")                                                              \
    X(int, steps, 512, "simulation steps to run")                                                                   \
    X(int, _step, 0, "current step")                                                                                \
    X(bool, wrap, true, "wrap the errosion from one side to the other (making result tileable)")                    \
    X(bool, _main_loop, true, "⛰️ run the main erosion loop (option to disable for testing)")                        \
    X(float, scale, 1.0, "[<0.0]: real world size of pixel, will make slopes more gradual")                         \
    X(float, min_height, -1000000.0, "[-∞,∞]: minimum height the terrain can erode down to")                        \
    X(float, max_height, 1000000.0, "[-∞,∞]: maximum height the terrain can erode down to")                         \
    X(float, rain_rate, 0.0, "[<0.0]: add water uniform or multiplied by rain map")                                 \
    X(float, flow_rate, 1.0, "[<0.0]: speed of the outflow, still capped by max_water_outflow and available water") \
    X(float, slope_jitter, 0.0, "[<0.0]: added jitter to the calculate slope values")                               \
    X(int, slope_jitter_mode, 0, "[0]: 32 bit hash for 4 values (fast); [1]: 4 x 32 bit hash")                      \
    X(float, max_water_outflow, 1000000.0, "[0,∞]: max outflow from a cell per a turn")                             \
    X(int, erosion_mode, 0, "erosion mode")                                                                         \
    X(float, erosion_rate, 0.0, "rate at which height becomes sediment based on water outflow")                     \
    X(float, sediment_yield, 0.0, "amount of sediment generated from erosion, set [0,1]")                           \
    X(float, sediment_capacity, 0.0, "sediment capacity of the water [0,1]")                                        \
    X(int, deposition_mode, 0, "UNUSED")                                                                            \
    X(float, deposition_threshold, 1000000.0, "deposit if outflow below threshold")                                 \
    X(float, deposition_rate, 0.0, "rate sediment becomes height or rock again, deposition_mode 0")                 \
    X(float, drain_rate, 0.0, "rate of water drain when reaching minimum height")                                   \
    X(int, evaporation_mode, 0, "0: basic; 1: shallow water quicker")                                               \
    X(float, evaporation_rate, 0.0, "speed at which water disappears")                                              \
    X(bool, sediment_layer_mode, false, "if active, store differing sediment types")                                \
    X(bool, sea_pass, false, "🌊 enable sea pass")                                                                  \
    X(float, sea_level, 0.0, "🌊 average sea level")                                                                \
    X(float, sea_tidal_range, 1.0, "🌊 mean tidal range")                                                           \
    X(bool, simple_collapse, false, "🏜️ simple gradient based collapse")                                            \
    X(float, simple_collapse_amount, 0.0, "🏜️ simple gradient based collapse")                                      \
    X(float, simple_collapse_threshold, 0.0, "🏜️ simple gradient based collapse")                                   \
    X(float, simple_collapse_yield, 1.0, "🏜️ simple gradient based collapse")                                       \
    X(float, simple_collapse_jitter, 0.0, "🏜️ simple gradient based collapse")                                      \
    X(float, wind_strength, 0.0, "🍃 NEW TEST ... wind simulation")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_DEBUG_OUTPUTS                                    \
    X(float, _debug_rain_total, 0.0, "tracking total rain")       \
    X(float, _debug_drain_total, 0.0, "tracking total drain")     \
    X(float, _debug_erosion_total, 0.0, "tracking total erosion") \
    X(float, _debug_evaporation_total, 0.0, "tracking total erosion")

// (TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAYS                                                                         \
    X(float, 2, 1, height_map, "current height, set this map to start the simulation")                       \
    X(float, 2, 1, water_map, "current water, optionally set this map at start")                             \
    X(float, 2, 1, _water_out, "current water, optionally set this map at start")                            \
    X(float, 2, 1, sediment_map, "current sediment,  optionally set this map at start")                      \
    X(float, 2, 1, _sediment_out, "current sediment,  optionally set this map at start")                     \
    X(float, 3, 8, _flux8, "8 water flow out to 8 neighbours")                                               \
    X(float, 3, 8, _sediment_flux8, "sediment flow out to 8 neighbours")                                     \
    X(float, 3, 2, _slope_vector2, "gradient vectors give slope direction and strength")                     \
    X(float, 2, 1, _slope_magnitude, "calculation of strength based on gradient vector")                     \
    X(float, 2, 1, _water_velocity, "🧪 scalar water velocity")                                              \
    X(float, 2, 1, rain_map, "optional rain map, multiply by this")                                          \
    X(float, 2, 1, hardness_map, "optional hardness map")                                                    \
    X(float, 3, 3, layer_map, "layered version of height_map, should be filled with 3 layers from RGB")      \
    X(float, 3, 3, sediment_layer_map, "optional storage of different sediment types")                       \
    X(float, 1, 1, layer_erosiveness, "array of erosion rate of layer (higher is faster)")                   \
    X(float, 1, 1, layer_yield, "array erosion rate of layer (higher is faster)")                            \
    X(float, 1, 1, layer_permeability, "❓ water drainage?")                                                 \
    X(float, 1, 1, layer_erosion_threshold, "❓ erosion rate of layer (higher is faster)")                   \
    X(float, 1, 1, layer_solubility, "array of sediment solubility of layer (if using sediment_layer_mode)") \
    X(int, 2, 1, _exposed_layer_map, "getting exposed layer")                                                \
    X(float, 2, 1, _sea_map, "the time from 0-1 a tile spends under the tide")                               \
    X(float, 3, 2, _wind_vector2, "wind directions for dust blowing")

// ================================================================ //

// standard pattern to expan a define to a "string" (with the quote marks)
#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

namespace TEMPLATE_NAMESPACE {

//
//
//
class TEMPLATE_CLASS_NAME; // forward declaration

class Stage {
  public:
    TEMPLATE_CLASS_NAME *parent = nullptr; // non-owning
    bool is_configured = false;

    Stage(TEMPLATE_CLASS_NAME *p) : parent(p) {}

    virtual ~Stage() = default;

    virtual void configure() = 0;
    virtual void process() = 0;
};
//
//
//

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
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    TYPE *NAME = nullptr;
    TEMPLATE_CLASS_DEVICE_ARRAYS
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

class TEMPLATE_CLASS_NAME : public template_e::TemplateE<Parameters> {

  protected:
    core::cuda::SyncedDeviceStruct<DebugOutputs> debug_outputs; // device side pars (new synced wrapper keeps a local copy)
    core::cuda::DeviceStruct<ArrayPtrs> dev_array_ptrs;         // device side pars

  public:
    // getter/setters for the pars
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION)           \
    TYPE get_##NAME() const { return _pars.host().NAME; } \
    void set_##NAME(TYPE value) { set_par(_pars.host().NAME, value); }
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
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    core::cuda::types::DeviceArray<TYPE, DIMENSIONS> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

// Alocate DeviceArray's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION)                             \
    void allocate_##NAME() {                                                     \
        if (height_map.empty()) throw std::runtime_error("height_map is empty"); \
        if (NAME.empty()) {                                                      \
            NAME.resize_helper(_pars.host()._width, _pars.host()._height, DIM3); \
            NAME.zero_device();                                                  \
        }                                                                        \
    }
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    // lazy function to return array pointers
    std::vector<core::cuda::types::DeviceArrayBase *> get_device_array_n_ptrs() override {
        if (_device_array_n_ptrs.empty()) {
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSION, DIM3, NAME, DESCRIPTION) \
    _device_array_n_ptrs.push_back(&NAME);
            TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif
        }
        return _device_array_n_ptrs;
    }

    core::cuda::CurandArray2D curand_array_2d;

    // get pointers to the arrays as a structure to send to device
    ArrayPtrs get_array_ptrs() {
        return {
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSION, DIM3, NAME, DESCRIPTION) \
    NAME.dev_ptr(),
            TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif
        };
    }

    // lazy static constant (cannot be changed)
    const std::vector<std::string> &get_array_names() {
        static const std::vector<std::string> names = {
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSION, DIM3, NAME, DESCRIPTION) #NAME,
            TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif
        };
        return names;
    }

    TEMPLATE_CLASS_NAME() {
        initialize();
        debug_outputs.set_stream(stream.get());
        dev_array_ptrs.set_stream(stream.get());
    }

    void allocate_device() override;
    void process() override;
    void _process1(); //
    void _process2();

    void debug_update();

    //
    //
    //

    // // recursive
    // constexpr uint32_t fnv1a(const char *str, uint32_t hash = 2166136261u) {
    //     return (*str) ? fnv1a(str + 1, (hash ^ uint32_t(*str)) * 16777619u) : hash;
    // }
    // non recursive for C++ 17
    constexpr uint32_t fnv1a(const char *str) {
        uint32_t hash = 2166136261u;
        for (size_t i = 0; str[i] != '\0'; ++i) {
            hash ^= static_cast<uint32_t>(str[i]);
            hash *= 16777619u;
        }
        return hash;
    }
    // constexpr uint32_t HeightMapReady = fnv1a("HeightMapReady"); // hash idea

    // constexpr uint32_t HeightMapReady = fnv1a("HeightMapReady");
    // constexpr uint32_t SlopesCalculated = fnv1a("SlopesCalculated");

    // std::unordered_set<uint32_t> flags;
    // flags.insert(HeightMapReady);

    // if (flags.count(SlopesCalculated)) {
    //     // already done
    // }

    //
    //
    //
// Stage Enumerations
// (NAME, DESCRIPTION)
#define TEMPLATE_CLASS_STAGES                                 \
    X(SIMPLE_COLLAPSE, "simple slope based collapse erosion") \
    X(CALCULATE_SLOPES, "calculate slopes")                   \
    X(RAIN, "")                                               \
    X(MAIN, "")                                               \
    X(SEA_PASS, "")

#ifdef TEMPLATE_CLASS_STAGES
    enum class Stage {
#define X(NAME, DESCRIPTION) \
    NAME,
        TEMPLATE_CLASS_STAGES
#undef X
    };
#define X(NAME, DESCRIPTION) \
    void STAGE_##NAME();
    TEMPLATE_CLASS_STAGES
#undef X

    void RUN_STAGE(Stage stage) {
        switch (stage) {
#define X(NAME, DESCRIPTION) \
    case Stage::NAME:        \
        STAGE_##NAME();      \
        break;
            TEMPLATE_CLASS_STAGES
#undef X
        }
    }
#endif

    //
    //
    //

    // mark if stage has run (so it has set it's memory up)
    // if(_stage_configured.count(stage))
    // _stage_configured.insert(stage);
    std::unordered_set<Stage> _stage_configured;

    //
    //
    //

    //
    //
    //
};

} // namespace TEMPLATE_NAMESPACE
