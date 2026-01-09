/*

🦑 GNC boilerplate 20260106 v1

dynamic properties for easy binding using CRTP and constexpr

⚠️ Same as first example, but with
    #include "gnc_boilerplate.cuh"

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Erosion
#define TEMPLATE_NAMESPACE gnc::erosion

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_1                                                                          \
    X(bool, _debug, false, "track certain information for monitoring")                                              \
    X(bool, debug_print, false, "print out information to console")                                                 \
    X(int, debug_mod, 1, "frequency to print the debug output")                                                     \
    X(int, _width, 512, "map width")                                                                                \
    X(int, _height, 512, "map height")                                                                              \
    X(int, _layer_count, 0, "layers for layer mode")                                                                \
    X(bool, _layer_mode, false, "layer mode detected")                                                              \
    X(int, _step, 0, "current step")                                                                                \
    X(bool, wrap, true, "wrap the errosion from one side to the other (making result tileable)")                    \
    X(float, scale, 1.0, "[<0.0]: real world size of pixel, will make slopes more gradual")                         \
    X(float, min_height, -1000000.0, "[-∞,∞]: minimum height the terrain can erode down to")                        \
    X(float, max_height, 1000000.0, "[-∞,∞]: maximum height the terrain can erode down to")                         \
    X(float, rain_rate, 0.0, "[<0.0]: add water uniform or multiplied by rain map")                                 \
    X(float, flow_rate, 1.0, "[<0.0]: speed of the outflow, still capped by max_water_outflow and available water") \
    X(float, slope_jitter, 0.0, "[<0.0]: added jitter to the calculate slope values")                               \
    X(int, slope_jitter_mode, 0, "[0]: 32 bit hash for 4 values (fast); [1]: 4 x 32 bit hash")                      \
    X(float, max_water_outflow, 1000000.0, "[0,∞]: max outflow from a cell per a turn")                             \
    X(float, sediment_capacity, 0.0, "sediment capacity of the water [0,1]")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_2                                                          \
    X(int, erosion_mode, 0, "erosion mode")                                                         \
    X(float, erosion_rate, 0.0, "rate at which height becomes sediment based on water outflow")     \
    X(float, sediment_yield, 0.0, "amount of sediment generated from erosion, set [0,1]")           \
    X(int, deposition_mode, 0, "UNUSED")                                                            \
    X(float, deposition_threshold, 1000000.0, "deposit if outflow below threshold")                 \
    X(float, deposition_rate, 0.0, "rate sediment becomes height or rock again, deposition_mode 0") \
    X(float, drain_rate, 0.0, "rate of water drain when reaching minimum height")                   \
    X(int, evaporation_mode, 0, "0: basic; 1: shallow water quicker")                               \
    X(float, evaporation_rate, 0.0, "speed at which water disappears")                              \
    X(bool, sediment_layer_mode, false, "if active, store differing sediment types")                \
    X(bool, sea_pass, false, "🌊 enable sea pass")                                                  \
    X(float, sea_level, 0.0, "🌊 average sea level")                                                \
    X(float, sea_tidal_range, 1.0, "🌊 mean tidal range")                                           \
    X(bool, simple_collapse, false, "🏜️ simple gradient based collapse")                            \
    X(float, simple_collapse_amount, 0.0, "🏜️ simple gradient based collapse")                      \
    X(float, simple_collapse_threshold, 0.0, "🏜️ simple gradient based collapse")                   \
    X(float, simple_collapse_yield, 1.0, "🏜️ simple gradient based collapse")                       \
    X(float, simple_collapse_jitter, 0.0, "🏜️ simple gradient based collapse")                      \
    X(float, wind_strength, 0.0, "🍃 NEW TEST ... wind simulation")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_1   \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_2

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS2                                                                                  \
    X(float, 2, height_map, "current height, set this map to start the simulation")                             \
    X(float, 2, water_map, "current water, optionally set this map at start")                                   \
    X(float, 2, _water_out, "current water, optionally set this map at start")                                  \
    X(float, 2, sediment_map, "current sediment,  optionally set this map at start")                            \
    X(float, 2, _sediment_out_map, "current sediment,  optionally set this map at start")                       \
    X(float, 3, _flux8_map, "8 water flow out to 8 neighbours")                                                 \
    X(float, 3, _sediment_flux8_map, "sediment flow out to 8 neighbours")                                       \
    X(float, 3, _slope_vector2_map, "gradient vectors give slope direction and strength")                       \
    X(float, 2, _slope_magnitude_map, "calculation of strength based on gradient vector")                       \
    X(float, 2, _water_velocity_map, "🧪 scalar water velocity")                                                \
    X(float, 2, rain_map, "optional rain map, multiply by this")                                                \
    X(float, 2, hardness_map, "optional hardness map")                                                          \
    X(float, 3, layer_map, "layered version of height_map, should be filled with 3 layers from RGB")            \
    X(float, 3, sediment_layer_map, "optional storage of different sediment types")                             \
    X(float, 1, layer_erosiveness_array, "array of erosion rate of layer (higher is faster)")                   \
    X(float, 1, layer_yield_array, "array erosion rate of layer (higher is faster)")                            \
    X(float, 1, layer_permeability_array, "❓ water drainage?")                                                 \
    X(float, 1, layer_erosion_threshold_array, "❓ erosion rate of layer (higher is faster)")                   \
    X(float, 1, layer_solubility_array, "array of sediment solubility of layer (if using sediment_layer_mode)") \
    X(int, 2, _exposed_layer_map, "getting exposed layer")                                                      \
    X(float, 2, _sea_map, "the time from 0-1 a tile spends under the tide")                                     \
    X(float, 3, _wind_vector2_map, "wind directions for dust blowing")

#include "gnc_boilerplate.cuh"