/*

dynamic properties for easy binding using CRTP and constexpr

*/
#pragma once

#include "gnc_base.cuh"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Example

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS        \
    X(bool, _debug, false, "")           \
    X(int, test_int, false, "")          \
    X(float, test_float, false, "")      \
    X(DeviceArrayFloat2D, input, {}, "") \
    X(DeviceArrayFloat2D, output, {}, "")

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS \
    X(bool, _debug, false, "")

// ================================================================================================================================
// [Boilerplate]
// --------------------------------------------------------------------------------------------------------------------------------

#ifndef TEMPLATE_CLASS_PARAMETERS
#error "TEMPLATE_CLASS_PARAMETERS must be defined before including this file"
#endif
#ifndef TEMPLATE_CLASS_NAME
#error "TEMPLATE_CLASS_NAME must be defined before including this file"
#endif

namespace gnc {

class TEMPLATE_CLASS_NAME : public GNC_Base<TEMPLATE_CLASS_NAME> {
    using Self = TEMPLATE_CLASS_NAME;

  private:
    int _dummy;

#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

  public:
    // --------------------------------------------------------------------------------------------------------------------------------
    static constexpr auto properties_impl() {
        return std::tuple{
        // macro expand to create tuple
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_PARAMETERS
#undef X
        };
    }
    // --------------------------------------------------------------------------------------------------------------------------------
    void process() override {

    };
};

} // namespace gnc

// undef defines
#undef TEMPLATE_CLASS_NAME
#undef TEMPLATE_CLASS_PARAMETERS
#undef TEMPLATE_CLASS_ARRAYS

