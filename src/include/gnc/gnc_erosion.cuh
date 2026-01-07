/*

🦑 GNC boilerplate 20260106 v1

dynamic properties for easy binding using CRTP and constexpr

⚠️ Same as first example, but with
    #include "gnc_boilerplate.cuh"

*/
#pragma once
#include "template_macro_undef.h" // guard from definesó

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Erosion
#define TEMPLATE_NAMESPACE gnc::erosion

// must be trivially_copyable
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS \
    X(bool, _debug, false, "")    \
    X(int, tile_size, false, "for chequer_test")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS                \
    X(DeviceArrayFloat2D, heightmap, {}, "") \
    X(DeviceArrayFloat3D, layermap, {}, "")

#include "gnc_boilerplate.cuh"