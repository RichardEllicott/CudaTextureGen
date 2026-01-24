/*

wind simulation, should blow sediment around

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Wind
#define TEMPLATE_NAMESPACE gnc::wind

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT                                            \
    X(int2, _size, {}, "")                                                      \
    X(int, _step, 0, "")                                                            \
    X(bool, wrap, true, "")                                                            \
    X(float, random_wind, 1.0f, "")                                                 \
    X(float, damp_wind, 0.01, "")                                                   \
    X(float, wind_influence, 0.1f, "[0, 1] ratio to exhange each step, keep small") \
    X(float, slope_influence, 0.1f, "[0, 1] influence of slopes")                   \
    X(float, wind_drag, 0.01f, "[0, 1]")                                            \
    X(float2, test_float2, {}, "test test_float2")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS                 \
    X(float, 2, height_map, "terrain height") \
    X(float, 2, dust_map, "dust")             \
    X(float, 3, wind_vec2_map, "wind")        \
    X(float, 3, wind_vec2_map_out, "wind")    \
    X(float, 3, slope_vec2_map, "slope calculations")

#include "gnc_boilerplate.cuh"

#undef DEFAULT_PERIOD