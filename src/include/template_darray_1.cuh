/*

🎃 DARRAY TEMPLATE 20251115-1

using new DeviceArray2D ... data is instantly uploaded and downloaded, no local copy

*/
#pragma once

// ================================================================ //
#include "template_macro_undef.h"
#define TEMPLATE_CLASS_NAME TemplateDArray1
#define TEMPLATE_NAMESPACE template_darray_1

// auto set up pars (added to python and to pars object for upload)
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                             \
    X(size_t, _width, 1024, "map width")                      \
    X(size_t, _height, 1024, "map height")                    \
    X(size_t, _block, 16, "block size (best to leave at 16)") \
    X(bool, test_bool, 0.0, "a test bool")                    \
    X(float, test_float, 0.0, "a test float")                 \
    X(int, test_int, 0.0, "a test int")

// DeviceArray2D ... abstraction of DeviceArray that will be visible in python
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_2DS \
    X(float, image, "testing DeviceArray2D")

// DeviceArray3D ... abstraction of DeviceArray that will be visible in python
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_3DS \
    X(float, device_array_3d, "testing DeviceArray3D")

// private DeviceArray's
// these can be multi-dimensional and are GPU side only
// (TYPE, DIMENSION, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAYS \
    X(float, 1, height_map_out, "second height buffer")

// methods  (⚠️ Experimental)
// these can be multi-dimensional and are GPU side only
// (TYPE, NAME, ...)
#define TEMPLATE_CLASS_METHODS \
    X(void, test_process)      \
    X(void, test_process2)

// DeviceArrayN ... new upgrade to DeviceArray
// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_NS                             \
    X(float, 2, device_array_n2d_test, "testing device array n2d") \
    X(float, 3, device_array_n3d_test, "testing device array n3d")

//

// ================================================================ //

#include "cuda_types.cuh"

namespace TEMPLATE_NAMESPACE {

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

// tracking vars for debug
#ifdef TEMPLATE_CLASS_DEBUG_DATA
struct DebugData {
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_DEBUG_DATA
#undef X
};
static_assert(std::is_trivially_copyable<DebugData>::value, "Parameters must remain trivially copyable for CUDA memcpy"); // optional
#endif

class TEMPLATE_CLASS_NAME {

    Parameters pars;                               // local pars
    core::cuda::DeviceStruct<Parameters> dev_pars; // device side pars

    dim3 block; // we now computer block in configure_device
    dim3 grid;  // and grid

    bool _device_configured = false;
    // call before launching a kernel to ensure pars are uploaded and block/grid calculated
    void configure_device() {
        if (!_device_configured) {
            dev_pars.upload(pars);
            block = dim3(pars._block, pars._block);
            grid = dim3((pars._width + block.x - 1) / block.x, (pars._height + block.y - 1) / block.y);
            _device_configured = true;
        }
    }

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
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    core::cuda::DeviceArray<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

// DeviceArray2D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION) \
    core::cuda::DeviceArray2D<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif

// DeviceArray3D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#define X(TYPE, NAME, DESCRIPTION) \
    core::cuda::DeviceArray3D<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#undef X
#endif

// DeviceArrayN's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    core::cuda::DeviceArrayN<TYPE, DIMENSIONS> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif

// Method's (⚠️ Experimental)
#ifdef TEMPLATE_CLASS_METHODS
#define X(TYPE, NAME) \
    TYPE NAME();
    TEMPLATE_CLASS_METHODS
#undef X
#endif

    TEMPLATE_CLASS_NAME() {

        // set all the streams (new feature mean uploading/downloading works on this objects stream)
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    NAME.set_stream(stream.get());
        TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION) \
    NAME.set_stream(stream.get());
        TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#define X(TYPE, NAME, DESCRIPTION) \
    NAME.set_stream(stream.get());
        TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#undef X
#endif

#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    NAME.set_stream(stream.get());
        TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif
    }

    //
    //
    //
    //
// 🚧 🚧 🚧 🚧
    // Reflection info if you want metadata
    struct ArrayInfo {
        core::cuda::DeviceArrayNBase *ptr;
        const char *description;
        int dimension;
    };

    // Lazy builder function
    inline std::vector<ArrayInfo> &all_arrays() {
        static std::vector<ArrayInfo> cache;
        if (cache.empty()) {
// Expand the X‑macro into initializer entries
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    cache.push_back({&NAME, DESCRIPTION, DIMENSION});
            TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
        }
        return cache;
    }

    //
    //
    //
    //
    //

    ~TEMPLATE_CLASS_NAME() {
    }

    void allocate_device();

    void deallocate_device() {

        // deallocate DeviceArray's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    NAME.free_device();
        TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

        // deallocate DeviceArray2D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION) \
    NAME.free_device();
        TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif

        // deallocate DeviceArray3D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#define X(TYPE, NAME, DESCRIPTION) \
    NAME.free_device();
        TEMPLATE_CLASS_DEVICE_ARRAY_3DS
#undef X
#endif

        // deallocate DeviceArrayN's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    NAME.free_device();
        TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X
#endif

//
// 
// 🚧 🚧 🚧 🚧
for (auto& info : all_arrays()) {
    info.ptr->free_device();
    std::cout << "Freed " << info.description
              << " (" << info.dimension << "D)\n";
}
//
//
//



        device_allocated = false;
    }

    void process();
};

} // namespace TEMPLATE_NAMESPACE
