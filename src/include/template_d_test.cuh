/*

🦑 TEMPLATE_D 20251130-1

2-part, trying constexpr

*/

#pragma once
#include "template_d_base.cuh"

// ================================================================ //
#define TEMPLATE_CLASS_NAME TemplateDTest
#define TEMPLATE_NAMESPACE template_d_test

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                             \
    X(size_t, _width, 1024, "map width")                      \
    X(size_t, _height, 1024, "map height")                    \
    X(size_t, _block, 16, "block size (best to leave at 16)") \
    X(bool, test_bool, 0.0, "a test bool")                    \
    X(float, test_float, 0.0, "a test float")                 \
    X(int, test_int, 0.0, "a test int")

// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_NS                             \
    X(float, 2, device_array_n2d_test, "testing device array n2d") \
    X(float, 3, device_array_n3d_test, "testing device array n3d") \
    X(float, 2, image, "new image")
// ================================================================ //

namespace TEMPLATE_NAMESPACE {

// Parameters struct for uploading to GPU
struct Parameters {
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif
};
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy"); // optional

class TEMPLATE_CLASS_NAME : public template_d::TemplateD<Parameters> {

  public:
    // getter/setters for the pars
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION)   \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { set_par(pars.NAME, value); }
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

// DeviceArray's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    core::cuda::types::DeviceArray<TYPE, DIMENSIONS> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif

    // lazy function to return array pointers
    std::vector<core::cuda::types::DeviceArrayBase *> get_device_array_n_ptrs() override {
        if (_device_array_n_ptrs.empty()) {
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    _device_array_n_ptrs.push_back(&NAME);
            TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif
        }
        return _device_array_n_ptrs;
    }

    TEMPLATE_CLASS_NAME() {
        initialize();
    }

    // ~TEMPLATE_CLASS_NAME() {

    //     deallocate_device();
    // }

    void allocate_device() override;

    void process() override;
};

} // namespace TEMPLATE_NAMESPACE
