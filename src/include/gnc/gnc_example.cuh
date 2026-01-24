/*

🦑 GNC boilerplate 20260106 v1

dynamic properties for easy binding using CRTP and constexpr

⚠️ Same as first example, but with
    #include "_gnc_boilerplate.cuh"

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Example
#define TEMPLATE_NAMESPACE gnc::example

#define TEMPLATE_CLASS_PARAMETERS_STRUCT_1 \
    X(bool, _debug, false, "")      \
    X(int, _width, 0, "")           \
    X(int, _height, 0, "")          \
    X(int, tile_size, 0, "for chequer_test")

#define TEMPLATE_CLASS_PARAMETERS_STRUCT_2 \
    X(bool, extra_test, false, "")

#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_1          \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_2


// --------------------------------------------------------------------------------------------------------------------------------

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS \
    X(float, 2, input, "")     \
    X(float, 2, output, "")



#include "_gnc_boilerplate.cuh"

#undef TEMPLATE_CLASS_PARAMETERS_STRUCT_1
#undef TEMPLATE_CLASS_PARAMETERS_STRUCT_2
#undef TEMPLATE_CLASS_PARAMETERS_STRUCT

#undef TEMPLATE_CLASS_ARRAYS