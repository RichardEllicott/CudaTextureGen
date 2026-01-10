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
#define TEMPLATE_CLASS_PARAMETERS_STRUCT         \
    X(int, _width, 256, "")                \
    X(int, _height, 256, "")               \

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS \
    X(float, 2, height_map, "terrain height")\
    X(float, 2, dust_map, "dust")\
    X(float, 3, wind_velocity2_map, "wind")\

#include "gnc_boilerplate.cuh"

#undef DEFAULT_PERIOD