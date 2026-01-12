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

#define DEFAULT_PERIOD {7, 7, 7}

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT         \
    X(bool, _debug, false, "")            \
    X(int, width, 256, "")                \
    X(int, height, 256, "")               \
    X(int, seed, 0, "")                   \
    X(float3, period, DEFAULT_PERIOD, "") \
    X(float3, offset, {}, "")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS \
    X(float, 2, output, "")

#include "gnc_boilerplate.cuh"

#undef DEFAULT_PERIOD