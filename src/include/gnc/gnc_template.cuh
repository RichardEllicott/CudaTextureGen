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
#define TEMPLATE_CLASS_NAME GNC_Template
#define TEMPLATE_NAMESPACE gnc::_template // ❗ template is reserved

// must be trivially_copyable
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                    \
    X(bool, _debug, false, "")                       \
    X(int, tile_size, false, "for chequer_test")     \
    X(FloatArray<8>, float8, {}, "float array test") \
    X(IntArray<8>, int8, {}, "int array test")

// more complicated objects typically Ref<> types
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS               \
    X(RefDeviceArrayFloat2D, input, {}, "") \
    X(RefDeviceArrayFloat2D, output, {}, "")

// different pattern for arrays, allows better introspection
// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS2 \
    X(float, 2, input2, "")    \
    X(float, 2, output2, "")

#pragma region BOILERPLATE
// ================================================================================================================================
// REFACTOR OPTIONS
// --------------------------------------------------------------------------------------------------------------------------------
#define REFACTOR_STORAGE_IN_PARS 0
// ================================================================================================================================
// [Boilerplate (all below can be cocidered a copy, should match)]
// --------------------------------------------------------------------------------------------------------------------------------
#include "gnc_base.cuh"

#ifndef TEMPLATE_CLASS_NAME
#error "TEMPLATE_CLASS_NAME must be defined before including this file"
#endif
#ifndef TEMPLATE_NAMESPACE
#error "TEMPLATE_NAMESPACE must be defined before including this file"
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
struct ArrayPointers {
#ifdef TEMPLATE_CLASS_ARRAYS2
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    TYPE *NAME;
    TEMPLATE_CLASS_ARRAYS2
#undef X
#endif
};
static_assert(std::is_trivially_copyable<ArrayPointers>::value, "ArrayPointers must remain trivially copyable for CUDA memcpy");
// ================================================================================================================================
// Main Class
// --------------------------------------------------------------------------------------------------------------------------------
class TEMPLATE_CLASS_NAME : public GNC_Base<TEMPLATE_CLASS_NAME, Parameters, ArrayPointers> {
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

    // bind arrays2 (second pattern)
#ifdef TEMPLATE_CLASS_ARRAYS2
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    core::Ref<core::cuda::DeviceArray<TYPE, DIMENSIONS>> NAME;
    TEMPLATE_CLASS_ARRAYS2
#undef X
#endif

  public:
    // --------------------------------------------------------------------------------------------------------------------------------
    static constexpr auto properties_impl() {
        return std::tuple{

#if REFACTOR_STORAGE_IN_PARS == 0

        // referencing class body (old version)
#ifdef TEMPLATE_CLASS_PARAMETERS // bind pars
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

#elif REFACTOR_STORAGE_IN_PARS == 1

        // NestedProperty<Self, &Self::parameters, &Parameters::tile_size>{"tile_size"}, // tested working

#ifdef TEMPLATE_CLASS_PARAMETERS // bind pars
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    NestedProperty<Self, &Self::parameters, &Parameters::NAME>{EXPAND_AND_STRINGIFY(NAME)},
            TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

#endif

#ifdef TEMPLATE_CLASS_ARRAYS // bind arrays
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
                TEMPLATE_CLASS_ARRAYS
#undef X
#endif

#ifdef TEMPLATE_CLASS_ARRAYS2 // bind arrays2 (second pattern)
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
                    TEMPLATE_CLASS_ARRAYS2
#undef X
#endif
        };
    }
    // --------------------------------------------------------------------------------------------------------------------------------

    static constexpr auto properties2_impl() {
        return std::tuple{};
    }
    // --------------------------------------------------------------------------------------------------------------------------------

    void ready_device_impl() {

        // copy all pars to struct
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    parameters.NAME = NAME;
        TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

        // copy all array pointers
#ifdef TEMPLATE_CLASS_ARRAYS2 // bind arrays2 (second pattern)
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    array_pointers.NAME = nullptr;             \
    if (NAME.is_valid()) array_pointers.NAME = NAME->dev_ptr();
        TEMPLATE_CLASS_ARRAYS2
#undef X
#endif
        // now upload the pars
        dev_parameters.upload(parameters);
        dev_array_pointers.upload(array_pointers);
    }
    // --------------------------------------------------------------------------------------------------------------------------------

    void process() override;
};

} // namespace TEMPLATE_NAMESPACE

#undef REFACTOR_STORAGE_IN_PARS

#pragma endregion
