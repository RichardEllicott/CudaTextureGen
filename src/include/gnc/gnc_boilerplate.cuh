//
// ⚠️ THIS FILE IS COPIED OR GENERATED FROM 'gnc_template.cuh'
//

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

    // ================================================================
    // [Create pars and arrays]
    // ----------------------------------------------------------------

    // create pars
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

    // create arrays
#ifdef TEMPLATE_CLASS_ARRAYS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_ARRAYS
#undef X
#endif

    // create arrays2 (second pattern)
#ifdef TEMPLATE_CLASS_ARRAYS2
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    core::Ref<core::cuda::DeviceArray<TYPE, DIMENSIONS>> NAME;
    TEMPLATE_CLASS_ARRAYS2
#undef X
#endif

  public:
    // ================================================================
    // [Properties Binding]
    // ----------------------------------------------------------------

    // CRTP requirement
    static constexpr auto _properties() {
        return std::tuple{

#if REFACTOR_GNC_STORAGE_IN_PARS == 0

        // referencing class body (old version)
#ifdef TEMPLATE_CLASS_PARAMETERS // bind pars
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

#elif REFACTOR_GNC_STORAGE_IN_PARS == 1

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
    // CRTP requirement
    static constexpr auto _properties2() {
        return std::tuple{

        };
    }

    // ================================================================
    // [Method Binding]
    // ----------------------------------------------------------------

    // CRTP requirement
    static constexpr auto _methods() {
        return std::tuple{
        // Method<Self, &Self::test_method2>{"test_method2"},

#ifdef TEMPLATE_CLASS_METHODS // bind methods
#define X(TYPE, NAME, DESCRIPTION) \
    Method<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME)},
            TEMPLATE_CLASS_METHODS
#undef X
#endif

        };
    }

    // --------------------------------------------------------------------------------------------------------------------------------

    // CRTP requirement
    void _ready_device() {

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
// bind extra methods
#ifdef TEMPLATE_CLASS_METHODS // bind arrays2 (second pattern)
#define X(TYPE, NAME, DESCRIPTION) \
    TYPE NAME();
    TEMPLATE_CLASS_METHODS
#undef X
#endif
    // --------------------------------------------------------------------------------------------------------------------------------

    void _compute(); // CRTP


};
} // namespace TEMPLATE_NAMESPACE

#undef REFACTOR_STORAGE_IN_PARS

#pragma endregion
