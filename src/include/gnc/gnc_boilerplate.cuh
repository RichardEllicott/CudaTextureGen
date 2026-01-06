
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
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy"); // optional
                                                                                                                           // ================================================================================================================================
// Main Class
// --------------------------------------------------------------------------------------------------------------------------------
class TEMPLATE_CLASS_NAME : public GNC_Base<TEMPLATE_CLASS_NAME> {
    using Self = TEMPLATE_CLASS_NAME;

    // bind properties
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
    TEMPLATE_CLASS_ARRAYS
#undef X

  public:
    // --------------------------------------------------------------------------------------------------------------------------------
    static constexpr auto properties_impl() {
        return std::tuple{
        // macro expand to create tuple
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_PARAMETERS
                TEMPLATE_CLASS_ARRAYS
#undef X
        };
    }
    // --------------------------------------------------------------------------------------------------------------------------------
    void process() override;
};

} // namespace TEMPLATE_NAMESPACE


