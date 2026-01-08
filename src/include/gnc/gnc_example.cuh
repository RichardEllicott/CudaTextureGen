/*

🦑 GNC boilerplate 20260106 v1

dynamic properties for easy binding using CRTP and constexpr

NOTES:

- region BOILERPLATE (is a the main boilerplate, it is also in gnc_boilerplate.cuh)

*/
#pragma once
#include "template_macro_undef.h" // guard from defines

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Example
#define TEMPLATE_NAMESPACE gnc::example

// must be trivially_copyable
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                    \
    X(bool, _debug, false, "")                       \
    X(int, tile_size, false, "for chequer_test")     \
    X(FloatArray<8>, float8, {}, "float array test") \
    X(IntArray<8>, int8, {}, "int array test")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS            \
    X(DeviceArrayFloat2D, input, {}, "") \
    X(DeviceArrayFloat2D, output, {}, "")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS2 \
    X(float, 2, input2, "")    \
    X(float, 2, output2, "")

#pragma region BOILERPLATE
// ================================================================================================================================
// [Boilerplate (all below can be cocidered a copy, should match)]
// --------------------------------------------------------------------------------------------------------------------------------
#include "gnc_base.cuh"

#ifndef TEMPLATE_CLASS_NAME
#error "TEMPLATE_CLASS_NAME must be defined before including this file"
#endif
#ifndef TEMPLATE_CLASS_ARRAYS
#error "TEMPLATE_CLASS_ARRAYS must be defined before including this file"
#endif
#ifndef TEMPLATE_CLASS_PARAMETERS
#error "TEMPLATE_CLASS_PARAMETERS must be defined before including this file"
#endif

namespace TEMPLATE_NAMESPACE {
// ================================================================================================================================
// Parameters struct for uploading to GPU (UNUSED)
// --------------------------------------------------------------------------------------------------------------------------------
struct Parameters {
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
};
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy");
// ================================================================================================================================
// ArrayDevicePointers struct for uploading to GPU (UNUSED)
// --------------------------------------------------------------------------------------------------------------------------------
struct ArrayDevicePointers {
#ifdef TEMPLATE_CLASS_ARRAYS2
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    TYPE *NAME;
    TEMPLATE_CLASS_ARRAYS2
#undef X
#endif
};
static_assert(std::is_trivially_copyable<ArrayDevicePointers>::value, "ArrayDevicePointers must remain trivially copyable for CUDA memcpy");
// ================================================================================================================================
// Main Class
// --------------------------------------------------------------------------------------------------------------------------------
class TEMPLATE_CLASS_NAME : public GNC_Base<TEMPLATE_CLASS_NAME, Parameters, ArrayDevicePointers> {
    using Self = TEMPLATE_CLASS_NAME;

    // bind pars
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

    // bind arrays
#ifdef TEMPLATE_CLASS_ARRAYS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_ARRAYS
#undef X
#endif

  public:
    // --------------------------------------------------------------------------------------------------------------------------------
    static constexpr auto properties_impl() {
        return std::tuple{

        // bind pars
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

        // bind arrays
#ifdef TEMPLATE_CLASS_ARRAYS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
                TEMPLATE_CLASS_ARRAYS
#undef X
#endif

        };
    }
    // --------------------------------------------------------------------------------------------------------------------------------

    static constexpr auto properties2_impl() {
        return std::tuple{};
    }
    // --------------------------------------------------------------------------------------------------------------------------------

    void process() override;
};

} // namespace TEMPLATE_NAMESPACE
#pragma endregion
