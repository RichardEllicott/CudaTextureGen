/*

erosion using new approach to sampling

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#define TEMPLATE_CLASS_NAME GNC_ErosionDelta
#define TEMPLATE_NAMESPACE gnc::erosion_delta
// ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

#define DEFAULT_BLOCK {16, 16}

#define FMAX 1000000.0f
#define FMIN -1000000.0f

#define EROSION_DEFAULT_LAYER_SETTINGS {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_1 \
    X(dim3, _block, DEFAULT_BLOCK, "")     \
    X(dim3, _grid, {}, "")                 \
    X(int2, _size, {}, "")                 \
    X(int, _step, 0, "")

#define TEMPLATE_CLASS_PARAMETERS_STRUCT_2 \
    X(int, steps, 1, "")                   \
    X(bool, wrap, true, "")                \
    X(float, slope_jitter, 0.0f, "[<0]")   \
    X(float, max_height, FMAX, "")         \
    X(float, min_height, FMIN, "")

// Water
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_3 \
    X(float, rain_rate, 0.0f, "")          \
    X(int, water_velocity_mode, 0, "")     \
    X(float, max_water_outflow, FMAX, "")  \
    X(float, drain_rate, 0.0f, "")         \
    X(float, evaporation_rate, 0.0f, "")

// Sediment
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_4                              \
    X(float, sediment_capacity, 0.0f, "[0,1]")                          \
    X(float, sediment_yield, 0.0f, "[0,1] sediment yield from erosion") \
    X(float, deposition_rate, 0.0f, "]")

// Erosion
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_5 \
    X(int, erosion_mode, 0, "[0]")         \
    X(float, erosion_rate, 0.0f, "[<0]")

// Layers
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_6                                        \
    X(int, _layer_count, 0, "")                                                   \
    X(FloatArray<8>, layer_erosiveness_array, EROSION_DEFAULT_LAYER_SETTINGS, "") \
    X(FloatArray<8>, layer_yield_array, EROSION_DEFAULT_LAYER_SETTINGS, "")

// #define TEMPLATE_CLASS_PARAMETERS_STRUCT_7
// #define TEMPLATE_CLASS_PARAMETERS_STRUCT_8

#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_1   \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_2   \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_3   \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_4   \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_5   \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_6

// TEMPLATE_CLASS_PARAMETERS_STRUCT_7
// TEMPLATE_CLASS_PARAMETERS_STRUCT_8

// ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS_STRUCT_1 \
    X(float, 2, height_map, "")        \
    X(float, 2, water_map, "")         \
    X(float, 2, sediment_map, "")

#define TEMPLATE_CLASS_ARRAYS_STRUCT_2 \
    X(float, 2, rain_map, "")

#define TEMPLATE_CLASS_ARRAYS_STRUCT_3 \
    X(float, 3, layer_map, "")         \
    X(int, 2, _exposed_layer_map, "")

// _slope_vector2_map

// //     X(float, 2, _water_vertical_velocity, "") \
// //     X(float, 3, _water_lateral_velocity, "")  \
// //     X(float, 3, _slope_vector2_map, "")

#define TEMPLATE_CLASS_ARRAYS_STRUCT_4                                           \
    X(float, 2, _water_velocity_map, "store water velocity")                     \
    X(float, 2, _water_out_map, "")                                              \
    X(float, 2, _sediment_out_map, "")                                           \
    X(float, 3, _slope_vector2_map, "slope direction (also encoding steepness)") \
    X(float, 3, _flux8_map, "")                                                  \
    X(float, 3, _sediment_flux8_map, "")

// #define TEMPLATE_CLASS_ARRAYS_STRUCT_3 \
//     X(int2, 1, _check_offsets, "")     \
//     X(float, 1, _check_distances, "")  \
//     X(float2, 1, _check_dot_vectors, "")

#define TEMPLATE_CLASS_ARRAYS_STRUCT \
    TEMPLATE_CLASS_ARRAYS_STRUCT_1   \
    TEMPLATE_CLASS_ARRAYS_STRUCT_2   \
    TEMPLATE_CLASS_ARRAYS_STRUCT_3   \
    TEMPLATE_CLASS_ARRAYS_STRUCT_4

// ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_METHODS               \
    X(void, test, "test method")             \
    X(void, setup, "")                       \
    X(void, GPU_calculate_slope_vectors, "") \
    X(void, GPU_calculate_layer, "")         \
    X(void, GPU_calculate_flux4, "")         \
    X(void, GPU_apply_flux4, "")

// ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

#include "_gnc_boilerplate.cuh"
