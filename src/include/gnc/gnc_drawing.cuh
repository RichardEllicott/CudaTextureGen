/*


*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Drawing
#define TEMPLATE_NAMESPACE gnc::drawing

#define DEFAULT_SIZE {128, 128}
#define DEFAULT_POSITION {64, 64}

#define TEMPLATE_CLASS_PARAMETERS_STRUCT \
    X(bool, _debug, false, "")           \
    X(int, mode, 0, "")                  \
    X(int2, size, DEFAULT_SIZE, "")      \
    X(float, radius, 32.0f, "")             \
    X(float2, position, DEFAULT_POSITION, "")

// --------------------------------------------------------------------------------------------------------------------------------

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS_STRUCT \
    X(float, 2, output, "")

// ================================================================================================================================

#include "_gnc_boilerplate.cuh"

#undef DEFAULT_SIZE
#undef DEFAULT_POSITION
