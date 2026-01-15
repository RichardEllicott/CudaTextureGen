/*

reample an image, distortion by a map

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================

#define TEMPLATE_CLASS_NAME GNC_Resample
#define TEMPLATE_NAMESPACE gnc::resample

#define DEFAULT_WARP_STRENGTH {1.0f, 1.0f}

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT                                                                            \
    X(int2, _size, {}, "map size")                                                                                  \
    X(bool, relative_offset, true, "relative offset warps relative, otherwise map would need absolute coordinates") \
    X(bool, scale_by_output_size, true, "scale works so input of 0.5 would be offset by half size of image ")       \
    X(int, sample_mode, 0, "UNUSED, bilinear only at the moment")

// DeviceArray2D ... abstraction of DeviceArray that will be visible in python
// // (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS                                               \
    X(float, 2, input, "input image")                                       \
    X(float, 2, output, "buffer array to write to")                         \
    X(float, 2, map_x, "image to offset x (feed with noise to warp image)") \
    X(float, 2, map_y, "image to offset y (feed with noise to warp image)")

// ================================================================================================================================

#include "gnc_boilerplate.cuh"

// #undef DEFAULT_PERIOD