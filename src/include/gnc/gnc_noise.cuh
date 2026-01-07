/*

🦑 GNC boilerplate 20260106 v1

dynamic properties for easy binding using CRTP and constexpr

*/
#pragma once
#include "template_macro_undef.h" // guard from defines

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Noise
#define TEMPLATE_NAMESPACE gnc::noise

#define DEFAULT_PERIOD {7, 7, 7}

// must be trivially_copyable
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS         \
    X(bool, _debug, false, "")            \
    X(int, seed, 0, "")                   \
    X(Float3, period, DEFAULT_PERIOD, "") \
    X(Float3, offset, {}, "")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS \
    X(DeviceArrayFloat2D, output, {}, "")

#include "gnc_boilerplate.cuh"

#undef DEFAULT_PERIOD