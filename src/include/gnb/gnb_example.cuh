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
// #include "core/cuda/types.cuh" // DeviceArray

#include "gnb_base.cuh"

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


    // constexpr tuple of descriptors pattern
    template <typename T>
struct Field {
    const char* name;
    T* ptr;  // pointer-to-member
};





// namespace nb = nanobind;

class TEMPLATE_CLASS_NAME : public GNB_Base {
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

    TEMPLATE_CLASS_NAME() {

//         //⚠️ WARNING WAS BAD!!!! set streams
// #ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
// #define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
//     NAME.set_stream(stream.get())                    \
//         TEMPLATE_CLASS_DEVICE_ARRAYS
// #undef X
// #endif
    }

    void process() override;
    // void process() override {
    // }
};

} // namespace TEMPLATE_NAMESPACE
