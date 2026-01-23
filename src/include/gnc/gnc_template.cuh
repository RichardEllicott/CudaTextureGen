/*

🧙‍♂️ gnc_template 20260115

contains master bolerplate that is copied via python script to gnc_boilerplate

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Template
#define TEMPLATE_NAMESPACE gnc::_template // ❗ template is reserved

// must be trivially_copyable
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT             \
    X(bool, _debug, false, "")                       \
    X(int2, _size, {}, "")                           \
    X(int, tile_size, false, "for chequer_test")     \
    X(FloatArray<8>, float8, {}, "float array test") \
    X(IntArray<8>, int8, {}, "int array test")

// properties will not be added to the struct, eg Ref<> types work
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS           \
    X(RefDeviceArrayFloat2D, input, {}, "") \
    X(RefDeviceArrayFloat2D, output, {}, "")

// different pattern for arrays, allows better introspection
// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS \
    X(float, 2, input2, "")   \
    X(float, 2, output2, "")

// optional class methods extra
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_METHODS \
    X(void, test, "test method")

#pragma region BOILERPLATE
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
// Parameters struct for uploading to GPU
// --------------------------------------------------------------------------------------------------------------------------------
struct Parameters {
    using Self = Parameters;

#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

    // reflection string to member functions
    static constexpr auto properties() {
        return std::tuple{
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT // bind pars
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif
        };
    }
};
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy");
// ================================================================================================================================
// ArrayDevicePointers struct for uploading to GPU (UNUSED)
// --------------------------------------------------------------------------------------------------------------------------------
struct ArrayPointers {
    using Self = ArrayPointers;

#ifdef TEMPLATE_CLASS_ARRAYS
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    TYPE *NAME;
    TEMPLATE_CLASS_ARRAYS
#undef X
#endif

    // reflection string to member functions ⚠️ seems to be broken for the arrays atm, can't reflect these yet!
    static constexpr auto properties() {
        return std::tuple{
#ifdef TEMPLATE_CLASS_ARRAYS // bind pars
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_ARRAYS
#undef X
#endif
        };
    }
};
static_assert(std::is_trivially_copyable<ArrayPointers>::value, "ArrayPointers must remain trivially copyable for CUDA memcpy");
// ================================================================================================================================
// Main Class
// --------------------------------------------------------------------------------------------------------------------------------
class TEMPLATE_CLASS_NAME : public GNC_Base<TEMPLATE_CLASS_NAME, Parameters, ArrayPointers> {
    using Self = TEMPLATE_CLASS_NAME;

    // ================================================================================================================================
    // [Create pars and arrays]
    // --------------------------------------------------------------------------------------------------------------------------------

    // create pars
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

    // create arrays
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

    // create arrays2 (second pattern)
#ifdef TEMPLATE_CLASS_ARRAYS
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    core::Ref<core::cuda::DeviceArray<TYPE, DIMENSIONS>> NAME;
    TEMPLATE_CLASS_ARRAYS
#undef X
#endif

// create methods
#ifdef TEMPLATE_CLASS_METHODS // bind arrays2 (second pattern)
#define X(TYPE, NAME, DESCRIPTION) \
    TYPE NAME();
    TEMPLATE_CLASS_METHODS
#undef X
#endif

  public:
    // ================================================================================================================================
    // [Properties Binding]
    // --------------------------------------------------------------------------------------------------------------------------------

    // CRTP requirement
    static constexpr auto _properties() {
        return std::tuple{

        // bind structure parameters
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

// bind non structure parameters
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
                TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

// bind arrays
#ifdef TEMPLATE_CLASS_ARRAYS
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
                    TEMPLATE_CLASS_ARRAYS
#undef X
#endif
        };
    }

    // ================================================================================================================================
    // [Method Binding]
    // --------------------------------------------------------------------------------------------------------------------------------

    // CRTP requirement
    static constexpr auto _methods() {
        return std::tuple{

// bind methods
#ifdef TEMPLATE_CLASS_METHODS
#define X(TYPE, NAME, DESCRIPTION) \
    Method<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME)},
            TEMPLATE_CLASS_METHODS
#undef X
#endif

        };
    }

    // ================================================================================================================================
    // [Ready Device]
    // --------------------------------------------------------------------------------------------------------------------------------

    // CRTP requirement
    void _ready_device() {

        // copy all pars to struct
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    _pars.NAME = NAME;
        TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

        // copy all array pointers
#ifdef TEMPLATE_CLASS_ARRAYS // bind arrays2 (second pattern)
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    _arrays.NAME = nullptr;                    \
    if (NAME.is_valid()) {                     \
        _arrays.NAME = NAME->dev_ptr();        \
    }
        TEMPLATE_CLASS_ARRAYS
#undef X
#endif
    }

    // ================================================================================================================================

    void _compute(); // CRTP requirement
};
} // namespace TEMPLATE_NAMESPACE

#undef REFACTOR_STORAGE_IN_PARS

#pragma endregion
