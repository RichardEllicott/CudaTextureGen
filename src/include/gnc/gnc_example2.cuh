/*

🦑 GNC boilerplate 20260106 v1

dynamic properties for easy binding using CRTP and constexpr

*/
#pragma once
#include "template_macro_undef.h" // guard from defines

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Example2
#define TEMPLATE_NAMESPACE gnc::example2

// must be trivially_copyable
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS \
    X(bool, _debug, false, "")    \
    X(int, tile_size, false, "for chequer_test")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS            \
    X(DeviceArrayFloat2D, input, {}, "") \
    X(DeviceArrayFloat2D, output, {}, "")

#include "gnc_boilerplate.cuh" // guard from defines
