/*


*/
#pragma once

#include "template_macro_undef.h"
// #include <nanobind/nanobind.h>

// #include <unordered_map>

// #include <memory> // shared_ptr
// #include <string>
// #include <unordered_map>

#include "core.h"
// #include "core/cuda/types_collection.cuh" // DeviceArray

#include "gnb_base.cuh"

#include "macros.h"

// ================================================================ //
#define TEMPLATE_CLASS_NAME GNB_Example
#define TEMPLATE_NAMESPACE gnb_example

// // (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
// #define TEMPLATE_CLASS_PARAMETERS \
//     X(bool, _debug, false, "")    \
//     X(int, test_int, false, "")   \
//     X(float, test_float, false, "")

// (TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAYS \
    X(float, 2, 1, input, "")        \
    X(float, 2, 1, output, "")

// ================================================================ //

namespace TEMPLATE_NAMESPACE {

template <typename T, typename Member>
struct Property {
    const char *name;
    Member member; // ✅ correct — Member *is already* a pointer-to-member
};

// namespace nb = nanobind;

class TEMPLATE_CLASS_NAME : public GNB_Base {
    using Self = TEMPLATE_CLASS_NAME;

  public:
    // ================================================================================================================================

    // Ref<DeviceArray>'s
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    core::Ref<core::cuda::DeviceArray<TYPE, DIMENSIONS>> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    // ================================================================================================================================
    float test_var = 777.0;

    // could put this on the child objects
    static constexpr auto properties() {
        return std::make_tuple(
        // Property<TEMPLATE_CLASS_NAME, decltype(&TEMPLATE_CLASS_NAME::input)>{"heightmap", &TEMPLATE_CLASS_NAME::input}
        // Property<TEMPLATE_CLASS_NAME, decltype(&TEMPLATE_CLASS_NAME::flow)>{"flow", &TEMPLATE_CLASS_NAME::flow}
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    Property<Self, decltype(&Self::NAME)>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif
                Property<Self, decltype(&Self::test_var)>{"test_var", &Self::test_var});
    }

    // ================================================================================================================================

    TEMPLATE_CLASS_NAME() {
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    NAME.instantiate();
        TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

        // stream.instantiate();
        // input->set_stream(stream.get().get());

        // ⚠️ WARNING WAS BAD!!!! set streams
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    NAME->set_stream(stream.get().get());            \
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif
    }

    // ================================================================================================================================

    void process() override;
    // void process() override {
    // }
};

} // namespace TEMPLATE_NAMESPACE
