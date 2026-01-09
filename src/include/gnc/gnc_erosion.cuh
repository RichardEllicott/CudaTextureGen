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
#define TEMPLATE_CLASS_PARAMETERS   \
    X(bool, _debug, false, "")      \
    X(bool, _layer_mode, false, "") \
    X(int, _layer_count, 0, "")     \
    X(int, _step, 0, "current step, used for hash calculations")     \
    X(float, jitter, 0.0f, "jitter for slope calculations")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS                                                 \
    X(RefDeviceArrayFloat2D, height_map, {}, "")                                 \
    X(RefDeviceArrayFloat2D, water_map, {}, "")                                 \
    X(RefDeviceArrayFloat2D, sediment_map, {}, "")                                 \
    X(RefDeviceArrayFloat3D, layer_map, {}, "")                                  \
    X(RefDeviceArrayInt2D, _exposed_layer_map, {}, "top exposed layer to erode") \
    X(RefDeviceArrayFloat3D, _slope_vector2_map, {}, "gradient vectors give slope direction and strength")

#include "gnc_boilerplate.cuh"