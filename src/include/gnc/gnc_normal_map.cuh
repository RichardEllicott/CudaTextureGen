/*

Normal Map Generator

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_NormalMap
#define TEMPLATE_NAMESPACE gnc::normal_map

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    X(int2, _size, {}, "")               \
    X(bool, wrap, true, "")              \
    X(bool, direct_x_style, true, "")    \
    X(float, normal_scale, 1.0, "")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS       \
    X(float, 2, input, "input map") \
    X(float, 3, output, "output map (rgb)")

#include "gnc_boilerplate.cuh"
