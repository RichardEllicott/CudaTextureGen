/*

🦑 GNC boilerplate 20260106 v1

dynamic properties for easy binding using CRTP and constexpr

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Noise
#define TEMPLATE_NAMESPACE gnc::noise
// --------------------------------------------------------------------------------------------------------------------------------
#define DEFAULT_PERIOD {7, 7, 7}
#define DEFAULT_SIZE {256, 256}
#define DEFAULT_WRAP {true, true, true}

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT  \
    X(bool, _debug, false, "")            \
    X(int2, size, DEFAULT_SIZE, "")       \
    X(int, seed, 0, "")                   \
    X(float3, period, DEFAULT_PERIOD, "") \
    X(float3, offset, {}, "")             \
    X(BoolArray<3>, wrap, DEFAULT_WRAP, "")

// #undef DEFAULT_PERIOD
// #undef DEFAULT_SIZE
// #undef DEFAULT_WRAP
// --------------------------------------------------------------------------------------------------------------------------------
// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS \
    X(float, 2, output, "")

#include "gnc_boilerplate.cuh"

