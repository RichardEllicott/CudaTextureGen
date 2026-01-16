/*



*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Erosion2
#define TEMPLATE_NAMESPACE gnc::erosion2

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_PRIVATE \
    X(int2, _size, {}, "")                       \
    X(int, _step, 0, "")                         \
    X(int, _layer_count, 0, "")                  \
    X(bool, _layer_mode, false, "")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_SETTINGS                                                                    \
    X(int, steps, 512, "")                                                                                           \
    X(bool, wrap, true, "")                                                                                          \
    X(float, slope_jitter, 0.0f, "")                                                                                 \
    X(float, flow_rate, 1.0f, "[<0.0]: speed of the outflow, still capped by max_water_outflow and available water") \
    X(float, max_water_outflow, 1000000.0, "[0,∞]: max outflow from a cell per a turn")                              \
    X(float, min_height, -1000000.0, "[-∞,∞]: minimum height the terrain can erode down to")                         \
    X(int, erosion_mode, 0, "erosion mode")                                                                          \
    X(float, erosion_rate, 0.0, "rate at which height becomes sediment based on water outflow")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_SEDIMENT \
    X(float, sediment_capacity, 0.0, "sediment capacity of the water [0,1]")

// --------------------------------------------------------------------------------------------------------------------------------

#define DEFAULT_LAYER_SETTINGS {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_LAYERS                           \
    X(FloatArray<8>, layer_erosiveness_array, DEFAULT_LAYER_SETTINGS, "") \
    X(FloatArray<8>, layer_yield_array, DEFAULT_LAYER_SETTINGS, "")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT      \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_PRIVATE  \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_SETTINGS \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_SEDIMENT \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_LAYERS

// --------------------------------------------------------------------------------------------------------------------------------

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS_1   \
    X(float, 2, height_map, "")   \
    X(float, 2, water_map, "")    \
    X(float, 2, sediment_map, "") \
    X(float, 2, rain_map, "")     \
    X(float, 3, layer_map, "")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS_2         \
    X(float, 2, _water_out_map, "")     \
    X(float, 2, _sediment_out_map, "")  \
    X(int, 2, _exposed_layer_map, "")   \
    X(float, 3, _slope_vector2_map, "") \
    X(float, 3, _flux8_map, "")         \
    X(float, 3, _sediment_flux8_map, "")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS \
    TEMPLATE_CLASS_ARRAYS_1   \
    TEMPLATE_CLASS_ARRAYS_2

// --------------------------------------------------------------------------------------------------------------------------------

// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_METHODS \
    X(void, setup, "setup stuff")

#include "gnc_boilerplate.cuh"