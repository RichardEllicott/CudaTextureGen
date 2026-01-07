/*

🎃 DARRAY TEMPLATE 20251115-1

using new DeviceArray2D ... data is instantly uploaded and downloaded, no local copy

*/
#pragma once

// ================================================================ //
#include "template_macro_undef.h"
#define TEMPLATE_CLASS_NAME Resample
#define TEMPLATE_NAMESPACE resample

// auto set up pars (added to python and to pars object for upload)
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                                                                                   \
    X(size_t, _width, 1024, "map width")                                                                            \
    X(size_t, _height, 1024, "map height")                                                                          \
    X(size_t, _block, 16, "block size (best to leave at 16)")                                                       \
    X(int, mode, 0, "0 = use the maps, 1 = rotate and offset (experimental)")                                       \
    X(bool, relative_offset, true, "relative offset warps relative, otherwise map would need absolute coordinates") \
    X(bool, scale_by_output_size, true, "scale works so input of 0.5 would be offset by half size of image ")       \
    X(float, warp_x_strength, 1.0, "optionally adjust map_x strength")                                              \
    X(float, warp_y_strength, 1.0, "optionally adjust map_y strength")                                              \
    X(float, angle, 0.0, "mode 1 rotate")                                                                           \
    X(float, offset_x, 0.0, "mode 1 offset x")                                                                      \
    X(float, offset_y, 0.0, "mode 1 offset x")                                                                      \
    X(int, sample_mode, 0, "UNUSED, bilinear only at the moment")

// DeviceArray2D ... abstraction of DeviceArray that will be visible in python
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_2DS                                  \
    X(float, input, "input image")                                       \
    X(float, output, "buffer array to write to")                         \
    X(float, map_x, "image to offset x (feed with noise to warp image)") \
    X(float, map_y, "image to offset y (feed with noise to warp image)")

// ================================================================ //

#include "core/cuda/types_collection.cuh"

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
    bool device_allocated = false;

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

// DeviceArray2D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION) \
    core::cuda::DeviceArray2D<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif

    void allocate_device();
    void deallocate_device();

    void process();
};

} // namespace TEMPLATE_NAMESPACE
