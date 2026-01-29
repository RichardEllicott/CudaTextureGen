/*

erosion using new approach to sampling

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
#define TEMPLATE_CLASS_NAME GNC_ErosionDelta
#define TEMPLATE_NAMESPACE gnc::erosion_delta
// // --------------------------------------------------------------------------------------------------------------------------------

#define DEFAULT_BLOCK {16, 16}

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT_1 \
    X(dim3, _block, DEFAULT_BLOCK, "")     \
    X(dim3, _grid, {}, "")                 \
    X(int2, _size, {}, "")                 \
    X(int, _step, 0, "")

#define TEMPLATE_CLASS_PARAMETERS_STRUCT_2 \
    X(int, steps, 1, "")                   \
    X(bool, wrap, true, "")                \
    X(float, slope_jitter, 0.0, "[<0]")

#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_1   \
    TEMPLATE_CLASS_PARAMETERS_STRUCT_2

// // // --------------------------------------------------------------------------------------------------------------------------------

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS_STRUCT_1 \
    X(float, 2, height_map, "")        \
    X(float, 2, water_map, "")         \
    X(float, 2, sediment_map, "")

#define TEMPLATE_CLASS_ARRAYS_STRUCT_2

// _slope_vector2_map

// //     X(float, 2, _water_vertical_velocity, "") \
// //     X(float, 3, _water_lateral_velocity, "")  \
// //     X(float, 3, _slope_vector2_map, "")

#define TEMPLATE_CLASS_ARRAYS_STRUCT_3  \
    X(float, 3, _slope_vector2_map, "") \
    X(float, 2, _slope_magnitude_map, "")

// #define TEMPLATE_CLASS_ARRAYS_STRUCT_3 \
//     X(int2, 1, _check_offsets, "")     \
//     X(float, 1, _check_distances, "")  \
//     X(float2, 1, _check_dot_vectors, "")

#define TEMPLATE_CLASS_ARRAYS_STRUCT \
    TEMPLATE_CLASS_ARRAYS_STRUCT_1   \
    TEMPLATE_CLASS_ARRAYS_STRUCT_2   \
    TEMPLATE_CLASS_ARRAYS_STRUCT_3

// // (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
// #define TEMPLATE_CLASS_PRIVATE_PARAMETERS \
//     X(dim3, block, DEFAULT_BLOCK, "")       \
//     X(dim3, grid, {}, "")

// // --------------------------------------------------------------------------------------------------------------------------------

// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_METHODS \
    X(void, test, "test method")\
    X(void, GPU_calculate_slope_vectors, "")\
    X(void, GPU_calculate_flux4, "")\
    X(void, GPU_apply_flux4, "")\
    





// ================================================================================================================================

#include "_gnc_boilerplate.cuh"
