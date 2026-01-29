/*

🎃 DARRAY TEMPLATE 20251115-1

attempting to achive rotation with 3D gradient noise

failed get this working

*/
#pragma once

// ================================================================ //
#include "template_macro_undef.h"
#define TEMPLATE_CLASS_NAME Noise
#define TEMPLATE_NAMESPACE noise

#include <array>

using Float9 = std::array<float, 9>; // for rotation matrix
using Float9Raw = float[9];          // For device structs

// auto set up pars (added to python and to pars object for upload)
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                                                           \
    X(size_t, width, 1024, "map width")                                                     \
    X(size_t, height, 1024, "map height")                                                   \
    X(size_t, _block, 16, "block size (best to leave at 16)")                               \
    X(int, type, 0, "")                                                                     \
    X(int, seed, 0, "")                                                                     \
    X(float, period, 7.0f, "the frequency of the noise, use no fraction to get seamless")   \
    X(float, period_x, 7.0f, "the frequency of the noise, use no fraction to get seamless") \
    X(float, period_y, 7.0f, "the frequency of the noise, use no fraction to get seamless") \
    X(float, period_z, 7.0f, "the frequency of the noise, use no fraction to get seamless") \
    X(float, _scale, 1.0f, "gets calculated")                                               \
    X(float, _scale_x, 1.0f, "gets calculated")                                             \
    X(float, _scale_y, 1.0f, "gets calculated")                                             \
    X(float, _scale_z, 1.0f, "gets calculated")                                             \
    X(float, x, 0.0f, "x position")                                                         \
    X(float, y, 0.0f, "y position")                                                         \
    X(float, z, 0.0f, "z position (for 3D noise)")                                          \
    X(bool, wrap_x, true, "")                                                               \
    X(bool, wrap_y, true, "")                                                               \
    X(bool, wrap_z, true, "")                                                               \
    X(float, rotate_x, 0.0f, "pitch")                                                       \
    X(float, rotate_y, 0.0f, "yaw")                                                         \
    X(float, rotate_z, 0.0f, "roll (the normal one we use for 2D face on images)")

// X(Float9, _rotation_matrix, {}, "calculated rotation matrix")

// #define NOISE_GENERATOR_PARAMETERS \
//     X(int, type, 0)                \
//     X(int, seed, 0)                \
//     X(float, period, 7.0f)         \
//     X(float, scale, 1.0f)          \
//     X(float, x, 0.0f)              \
//     X(float, y, 0.0f)              \
//     X(float, z, 0.0f)              \
//     X(float, warp_amp, 4.0f)       \
//     X(float, warp_scale, 1.0f)     \
//     X(float, angle, 0.0f)

// DeviceArray2D ... abstraction of DeviceArray that will be visible in python
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_2DS \
    X(float, noise, "noise")

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

    // Float9Raw _rotation_matrix;
    float _rotation_matrix[9]; // Direct declaration, no typedef
};
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy"); // optional
#endif

class TEMPLATE_CLASS_NAME {

    Parameters pars;                               // local pars
    core::cuda::DeviceStruct<Parameters> dev_pars; // device side pars

    dim3 block;
    dim3 grid;

    bool _refresh_device_config = false;
    // call before launching a kernel to ensure pars are uploaded and block/grid calculated
    void refresh_device_config() {
        if (!_refresh_device_config) {
            dev_pars.upload(pars);
            block = dim3(pars._block, pars._block);
            grid = dim3((pars.width + block.x - 1) / block.x, (pars.height + block.y - 1) / block.y);
            _refresh_device_config = true;
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
            _refresh_device_config = false;       \
        pars.NAME = value;                        \
    }
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

// DeviceArray2D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION) \
    core::cuda::types::DeviceArray2D<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif

    void allocate_device();
    void deallocate_device();

    void process();
};

} // namespace TEMPLATE_NAMESPACE
