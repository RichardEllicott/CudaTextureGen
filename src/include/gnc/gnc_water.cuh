/*

water simulation with velocity

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Water
#define TEMPLATE_NAMESPACE gnc::water

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    X(int2, _size, {}, "")               \
    X(int, _step, 0, "")                 \
    X(int, steps, 1, "")                 \
    X(bool, wrap, true, "")              \
    X(float, slope_jitter, 0.0f, "")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS_STRUCT          \
    X(float, 2, height_map, "")               \
    X(float, 2, water_map, "")                \
    X(float, 2, sediment_map, "")             \
    X(float, 2, _water_vertical_velocity, "") \
    X(float, 3, _water_lateral_velocity, "")  \
    X(float, 3, _slope_vector2_map, "")

// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_METHODS \
    X(void, test, "test method")

#include "_gnc_boilerplate.cuh"
