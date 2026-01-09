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

// must be trivially_copyable
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS \
    X(bool, _debug, false, "")    \
    X(int, tile_size, false, "for chequer_test")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS            \
    X(RefDeviceArrayFloat2D, input, {}, "") \
    X(RefDeviceArrayFloat2D, output, {}, "")

#include "gnc_boilerplate.cuh"