/*

CURRENT PORTING MODEL

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Erosion
#define TEMPLATE_NAMESPACE gnc::erosion

#define EROSION_DEFAULT_LAYER_SETTINGS {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}

// --------------------------------------------------------------------------------------------------------------------------------

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_1 \
    X(int2, _size, {}, "")                 \
    X(int, _step, 0, "")                   \
    X(int, _layer_count, 0, "")            \
    X(bool, _layer_mode_enabled, false, "")

#define TEMPLATE_CLASS_PARAMETERS_STRUCT_2                                                                        \
    X(int, steps, 512, "")                                                                                        \
    X(bool, wrap, true, "wrap the map (tileable)")                                                                \
    X(float, slope_jitter, 0.0, "[<0]")                                                                           \
    X(int, slope_jitter_mode, 0, "[0] 32 bit hash for 4 values (fast); [1]: 4 x 32 bit hash")                     \
    X(float, scale, 1.0, "[<0] real world size of pixel, will make slopes more gradual")                          \
    X(float, flow_rate, 1.0, "[<0]: speed of the outflow, still capped by max_water_outflow and available water") \
    X(float, max_water_outflow, 1000000.0, "[0,∞]: max outflow from a cell per a turn")                           \
    X(float, sediment_capacity, 0.0, "sediment capacity of the water [0,1]")                                      \
    X(float, min_height, -1000000.0, "[-∞,∞]: minimum height the terrain can erode down to")                      \
    X(float, max_height, 1000000.0, "[-∞,∞]: maximum height the terrain can erode down to")                       \
    X(bool, sediment_layer_mode, false, "if active, store differing sediment types")                              \
    X(float, sediment_yield, 0.0, "amount of sediment generated from erosion, set [0,1]")                         \
    X(float, deposition_threshold, 1000000.0, "deposit if outflow below threshold")                               \
    X(float, deposition_rate, 0.0, "rate sediment becomes height or rock again, deposition_mode 0")               \
    X(float, drain_rate, 0.0, "rate of water drain when reaching minimum height")                                 \
    X(int, evaporation_mode, 0, "0: basic; 1: shallow water quicker")                                             \
    X(float, evaporation_rate, 0.0, "speed at which water disappears")

// erosion
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_3  \
    X(int, erosion_mode, 0, "erosion mode") \
    X(float, erosion_rate, 0.0, "rate at which height becomes sediment based on water outflow")

// layer arrays

// ⚠️ THESE ARRAYS ARE NOT ACCESIBLE IN CUDA!!!
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_4                                        \
    X(FloatArray<8>, layer_erosiveness_array, EROSION_DEFAULT_LAYER_SETTINGS, "") \
    X(FloatArray<8>, layer_yield_array, EROSION_DEFAULT_LAYER_SETTINGS, "")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_1   \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_2   \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_3   \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_4

// --------------------------------------------------------------------------------------------------------------------------------

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS_1   \
    X(float, 2, height_map, "")   \
    X(float, 2, water_map, "")    \
    X(float, 2, sediment_map, "") \
    X(float, 2, rain_map, "")     \
    X(float, 3, layer_map, "")

#define TEMPLATE_CLASS_ARRAYS_2         \
    X(float, 2, _water_out_map, "")     \
    X(float, 2, _sediment_out_map, "")  \
    X(int, 2, _exposed_layer_map, "")   \
    X(float, 3, _slope_vector2_map, "") \
    X(float, 3, _flux8_map, "")         \
    X(float, 3, _sediment_flux8_map, "")

#define TEMPLATE_CLASS_ARRAYS_3                                                       \
    X(float, 2, _slope_magnitude_map, "calculation of strength based on gradient vector") \
    X(float, 2, _water_velocity_map, "🧪 scalar water velocity")

#define TEMPLATE_CLASS_ARRAYS \
    TEMPLATE_CLASS_ARRAYS_1   \
    TEMPLATE_CLASS_ARRAYS_2   \
    TEMPLATE_CLASS_ARRAYS_3

// #define TEMPLATE_CLASS_ARRAYS     \
//     EVAL(TEMPLATE_CLASS_ARRAYS_1) \
//     EVAL(TEMPLATE_CLASS_ARRAYS_2) \
//     EVAL(TEMPLATE_CLASS_ARRAYS_3)

// --------------------------------------------------------------------------------------------------------------------------------

// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_METHODS \
    X(void, setup, "setup stuff")

#include "gnc_boilerplate.cuh"

// cleanup loose defines
#undef TEMPLATE_CLASS_ARRAYS_1
#undef TEMPLATE_CLASS_ARRAYS_2
#undef TEMPLATE_CLASS_ARRAYS_3
