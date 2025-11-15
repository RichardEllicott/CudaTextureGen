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
    X(bool, test_bool, 0.0, "test bool")                      \
    X(float, test_float, 0.0, "test float")                   \
    X(int, test_int, 0.0, "test int")

// DeviceArray2D ... abstraction of DeviceArray that will be visible in python
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_2DS \
    X(float, device_array_2d, "testing DeviceArray2D")


// DeviceArray2D ... abstraction of DeviceArray that will be visible in python
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_3DS \
    X(float, device_array_3d, "testing DeviceArray3D")

// private DeviceArray's
// these can be multi-dimensional and are GPU side only
// (TYPE, DIMENSION, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAYS                        \
    X(float, 1, height_map_out, "second height buffer")     \
    X(float, 1, water_map_out, "second water buffer")       \
    X(float, 1, sediment_map_out, "second sediment buffer") \
    X(float, 8, flux8, "water flow out to 8 neighbours")

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

class TEMPLATE_CLASS_NAME {

    Parameters pars;                               // local pars
    core::cuda::DeviceStruct<Parameters> dev_pars; // device side pars

    bool pars_synced = false; // record if the pars have been synced since update
    void sync_pars() {
        if (!pars_synced) {
            dev_pars.upload(pars);
            pars_synced = true;
        }
    }

    core::cuda::Stream stream; // will be allocated along with object

  public:
    // getter/setters for the pars
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION)   \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) {                 \
        if (pars.NAME != value)                   \
            pars_synced = false;                  \
        pars.NAME = value;                        \
    }
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


    void process();
};

} // namespace TEMPLATE_NAMESPACE
