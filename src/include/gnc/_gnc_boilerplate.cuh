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

    // constexpr binding for reflection
    static constexpr auto properties() {
        return std::tuple{
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT // bind pars
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{#NAME, &Self::NAME},
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

#ifdef TEMPLATE_CLASS_ARRAYS_STRUCT
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    TYPE *NAME;
    TEMPLATE_CLASS_ARRAYS_STRUCT
#undef X
#endif

    // constexpr binding for reflection
    static constexpr auto properties() {
        return std::tuple{
#ifdef TEMPLATE_CLASS_ARRAYS_STRUCT // bind pars
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    Property<Self, &Self::NAME>{#NAME, &Self::NAME},
            TEMPLATE_CLASS_ARRAYS_STRUCT
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

    // ADDING new rand array

  public:




    // ================================================================================================================================
    // [Create Pars, Arrays and Methods]
    // --------------------------------------------------------------------------------------------------------------------------------

    // create pars (mirrored on struct)
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

    // create pars (no struct)
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

    // create array ref's
#ifdef TEMPLATE_CLASS_ARRAYS_STRUCT
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    core::Ref<core::cuda::types::DeviceArray<TYPE, DIMENSIONS>> NAME;
    TEMPLATE_CLASS_ARRAYS_STRUCT
#undef X
#endif

// private paramaters
#ifdef TEMPLATE_CLASS_PRIVATE_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PRIVATE_PARAMETERS
#undef X
#endif

// create methods
#ifdef TEMPLATE_CLASS_METHODS // bind arrays2 (second pattern)
#define X(TYPE, NAME, DESCRIPTION) \
    TYPE NAME();
    TEMPLATE_CLASS_METHODS
#undef X
#endif

    // ================================================================================================================================
    // [Properties Binding]
    // --------------------------------------------------------------------------------------------------------------------------------

    // CRTP requirement
    static constexpr auto _properties() {
        return std::tuple{

        // bind structure parameters
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{#NAME, &Self::NAME},
            TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

// bind non structure parameters
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{#NAME, &Self::NAME},
                TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

// bind arrays
#ifdef TEMPLATE_CLASS_ARRAYS_STRUCT
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    Property<Self, &Self::NAME>{#NAME, &Self::NAME},
                    TEMPLATE_CLASS_ARRAYS_STRUCT
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
    Method<Self, &Self::NAME>{#NAME},
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

        // functionality of this section being replaced by constexpr voodoo

// #define MACRO_COPY_METHOD
#ifdef MACRO_COPY_METHOD

        // copy all pars to struct
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    _pars.NAME = NAME;
        TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

        // copy all array pointers
#ifdef TEMPLATE_CLASS_ARRAYS_STRUCT // bind arrays2 (second pattern)
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    _arrays.NAME = nullptr;                    \
    if (NAME.is_valid()) {                     \
        _arrays.NAME = NAME->dev_ptr();        \
    }
        TEMPLATE_CLASS_ARRAYS_STRUCT
#undef X
#endif

#endif
    }

    // ================================================================================================================================

    void _compute(); // CRTP requirement
};
} // namespace TEMPLATE_NAMESPACE

#undef REFACTOR_STORAGE_IN_PARS

#pragma endregion
