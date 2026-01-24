/*

Ambient Occlusion Generator

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"
// ================================================================================================================================
#define TEMPLATE_CLASS_NAME GNC_AO_Map
#define TEMPLATE_NAMESPACE gnc::ao_map

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    X(int2, _size, {}, "")               \
    X(bool, wrap, true, "")              \
    X(int, mode, 0, "")                  \
    X(int, radius, 1, "")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS       \
    X(float, 2, input, "input map") \
    X(float, 2, output, "output map")

// ================================================================================================================================
#include "_gnc_boilerplate.cuh"
