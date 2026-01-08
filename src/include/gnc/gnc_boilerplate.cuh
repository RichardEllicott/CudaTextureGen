
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
