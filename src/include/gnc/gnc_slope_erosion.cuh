/*

simple slope erosion

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_SlopeErosion
#define TEMPLATE_NAMESPACE gnc::slope_erosion

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    X(int2, _size, {}, "")               \
    X(int, _step, 0, "")                 \
    X(int, steps, 512, "")               \
    X(bool, wrap, true, "")              \
    X(float, slope_threshold, 0.0f, "")  \
    X(float, erosion_rate, 0.0f, "")     \
    X(float, deposition_rate, 0.0f, "")  \
    X(float, jitter, 0.0f, "")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS   \
    X(float, 2, height_map, "") \
    X(float, 2, sediment_map, "")

// --------------------------------------------------------------------------------------------------------------------------------


#include "gnc_boilerplate.cuh"

