/*

🦑 DARRAY TEMPLATE 20251125-1

using new DeviceArrayN, multidimensional template with common DeviceArrayNBase


ATTEMPTING TO USE EXTREME POLYMORPHISM TO USE LESS MACROS

*/
#pragma once

// ================================================================ //
#include "template_macro_undef.h"
#define TEMPLATE_CLASS_NAME TemplateDArray2
#define TEMPLATE_NAMESPACE template_darray_2

// auto set up pars (added to python and to pars object for upload)
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                             \
    X(size_t, _width, 1024, "map width")                      \
    X(size_t, _height, 1024, "map height")                    \
    X(size_t, _block, 16, "block size (best to leave at 16)") \
    X(bool, test_bool, 0.0, "a test bool")                    \
    X(float, test_float, 0.0, "a test float")                 \
    X(int, test_int, 0.0, "a test int")

// DeviceArrayN ... new upgrade to DeviceArray
// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_NS                             \
    X(float, 2, device_array_n2d_test, "testing device array n2d") \
    X(float, 3, device_array_n3d_test, "testing device array n3d") \
    X(float, 2, image, "new image")

//

// ================================================================ //

// standard pattern to expan a define to a "string" (with the quote marks)
#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

#include "cuda_types.cuh"
#include <iostream>
#include <string> // not sure?
#include <typeindex>
#include <vector>

namespace TEMPLATE_NAMESPACE {

/* CRTP example
template <typename Derived>
class CRTP_Base {
public:
void call_required() {
    // Forward to derived — must exist!
    static_cast<Derived *>(this)->required_fn();
}
};

// Derived class must implement required_fn
class CRTP_Test : public CRTP_Base<CRTP_Test> {
public:
void required_fn() {
    std::cout << "MyArray::required_fn implemented\n";
}
};
*/

// testing making a derived class
class Base {

  protected:
    bool _device_configured = false;

    // template called by xmacro (just to reduce code in xmacro)
    template <typename T>
    void set_par(T &field, const T &value) {
        if (field != value) {
            _device_configured = false;
            field = value;
        }
    }

    core::cuda::Stream stream;     // will be allocated along with object
    bool device_allocated = false; // use or not, not sure but marking if the maps are uploaded

    dim3 block; // we now computer block in configure_device
    dim3 grid;  // and grid

    // virtual void configure_device() = 0;

    std::vector<core::cuda::DeviceArrayNBase *> _device_array_n_ptrs; // store pointers to the DeviceArrayN's for reflection

    // needs to be a virtual function as we need to use a macro to fill _device_array_n_ptrs
    virtual std::vector<core::cuda::DeviceArrayNBase *> get_device_array_n_ptrs() = 0;

    void deallocate_device() {
        for (auto &ptr : get_device_array_n_ptrs()) {
            ptr->free_device();
        }
        device_allocated = false;
    }

    virtual void configure_device() = 0;
    virtual void allocate_device() = 0;
    virtual void process() = 0;

#pragma region NEW_METADATA
    struct DeviceArrayNInfo {
        std::type_index type;    // element type (from typeid)
        int dimension;           // dimension metadata
        const char *name;        // name
        const char *description; // human-readable description
    };

    // Lazy builder function
    std::vector<DeviceArrayNInfo> &device_array_n_info() {
        static std::vector<DeviceArrayNInfo> _device_array_n_info;
        if (_device_array_n_info.empty()) {
// Expand the X‑macro into initializer entries
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    _device_array_n_info.push_back({typeid(TYPE), DIMENSION, EXPAND_AND_STRINGIFY(NAME), DESCRIPTION});
            TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
        }
        return _device_array_n_info;
    }
#pragma endregion
};

template <typename ParametersT>
class BaseTemplate : public Base {

  protected:
    ParametersT pars;                               // local pars
    core::cuda::DeviceStruct<ParametersT> dev_pars; // device side pars

    // call before launching a kernel to ensure pars are uploaded and block/grid calculated
    void configure_device() override {
        if (!_device_configured) {
            dev_pars.upload(pars);
            block = dim3(pars._block, pars._block);
            grid = dim3((pars._width + block.x - 1) / block.x, (pars._height + block.y - 1) / block.y);
            _device_configured = true;
        }
    }
};

// Parameters struct for uploading to GPU
#ifdef TEMPLATE_CLASS_PARAMETERS
struct Parameters {
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
};
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy"); // optional
#endif

class TEMPLATE_CLASS_NAME : public BaseTemplate<Parameters> {

  public:
    // getter/setters for the pars
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION)   \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { set_par(pars.NAME, value); }
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

// DeviceArrayN's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    core::cuda::DeviceArrayN<TYPE, DIMENSIONS> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif

    // lazy function to return array pointers
    std::vector<core::cuda::DeviceArrayNBase *> get_device_array_n_ptrs() override {
        if (_device_array_n_ptrs.empty()) {
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    NAME.set_stream(stream.get());            \
    _device_array_n_ptrs.push_back(&NAME);
            TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif
        }
        return _device_array_n_ptrs;
    }

    // TEMPLATE_CLASS_NAME() {
    // }

    // ~TEMPLATE_CLASS_NAME() {

    //     deallocate_device();
    // }

    void allocate_device() override;

    void process() override;
};

} // namespace TEMPLATE_NAMESPACE
