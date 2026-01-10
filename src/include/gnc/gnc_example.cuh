/*

🦑 GNC boilerplate 20260106 v1

dynamic properties for easy binding using CRTP and constexpr

⚠️ Same as first example, but with
    #include "gnc_boilerplate.cuh"

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Example
#define TEMPLATE_NAMESPACE gnc::example

// SUCESS!11
#define TEMPLATE_CLASS_PARAMETERS_1 \
    X(bool, _debug, false, "")      \
    X(int, _width, 0, "")           \
    X(int, _height, 0, "")          \
    X(int, tile_size, 0, "for chequer_test")

#define TEMPLATE_CLASS_PARAMETERS_2 \
    X(bool, extra_test, false, "")

#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    TEMPLATE_CLASS_PARAMETERS_1          \
    TEMPLATE_CLASS_PARAMETERS_2

// --------------------------------------------------------------------------------------------------------------------------------

// // BROKEN ... NVCC will not expand early

// // Force expansion with an extra level
// #define EXPAND_PARAMS(a, b) a b

// #define TEMPLATE_CLASS_PARAMETERS_STRUCT \
//     EXPAND_PARAMS(TEMPLATE_CLASS_PARAMETERS_1, TEMPLATE_CLASS_PARAMETERS_2)

// // // // Now you can undef the sub-lists
// // #undef TEMPLATE_CLASS_PARAMETERS_1
// #undef TEMPLATE_CLASS_PARAMETERS_2

// --------------------------------------------------------------------------------------------------------------------------------


// NVCC will not expand early


// #define EVAL(...) __VA_ARGS__
// #define EXPAND(...) EVAL(EVAL(__VA_ARGS__))

// #define TEMPLATE_CLASS_PARAMETERS \
//     EXPAND(TEMPLATE_CLASS_PARAMETERS_1 TEMPLATE_CLASS_PARAMETERS_2)

// // Now safe to undef
// // #undef TEMPLATE_CLASS_PARAMETERS_1
// // #undef TEMPLATE_CLASS_PARAMETERS_2

// --------------------------------------------------------------------------------------------------------------------------------

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS \
    X(float, 2, input, "")     \
    X(float, 2, output, "")

#

#include "gnc_boilerplate.cuh"