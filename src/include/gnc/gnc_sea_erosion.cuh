/*

sea erosion simulation

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_SeaErosion
#define TEMPLATE_NAMESPACE gnc::sea_erosion

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    X(int2, _size, {}, "")               \
    X(int, _step, 0, "")                 \
    X(int, steps, 1, "")                 \
    X(bool, wrap, true, "")\


// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS   \
    X(float, 2, height_map, "") \
    X(float, 2, water_map, "")  \
    X(float, 2, sediment_map, "")\
    X(float, 2, _water_vertical_velocity, "")\

#include "_gnc_boilerplate.cuh"
