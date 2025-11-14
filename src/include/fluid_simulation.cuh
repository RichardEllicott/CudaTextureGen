/*

🧜‍♀️ TEMPLATE VERSION 20251109-1

this one uses more clever types and classes, needing less horrible macros

*/
#pragma once

// ================================================================ //
#define TEMPLATE_CLASS_NAME FluidSimulation
#define TEMPLATE_NAMESPACE fluid_simulation

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS \
    X(size_t, width, 1024, "")    \
    X(size_t, height, 1024, "")   \
    X(size_t, _block, 16, "")     \
    X(float, gravity, 9.8, "")    \
    X(float, dt, 1.0, "")         \
    X(float, cell_size, 1.0, "")  \
    X(float, frame_steps, 16, "") \
    X(float, frame_count, 8, "")  \
    X(float, wave_speed, 1.0, "")

// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_MAPS  \
    X(float, height_map, "") \
    X(float, water_map, "")

// (TYPE, DIMENSION, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAYS \
    X(float, 1, water_map_out, "second water map")

// NEW
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_2DS \
    X(float, device_array_2d_test, "testing device array")

// ================================================================ //

#include "cuda_types.cuh"

namespace TEMPLATE_NAMESPACE {

// Parameters struct for uploading to GPU
struct Parameters {
#ifdef TEMPLATE_CLASS_PARAMETERS
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

// could be used with kernels
struct MapPointers {
#ifdef TEMPLATE_CLASS_MAPS
#define X(TYPE, NAME, DESCRIPTION) \
    TYPE *NAME = nullptr;
    TEMPLATE_CLASS_MAPS
#undef X
#endif
};

class TEMPLATE_CLASS_NAME {

// make getter/setters for the pars
#ifdef TEMPLATE_CLASS_PARAMETERS
  private:
    Parameters pars;
    bool pars_synced = false; // record if the pars have been synced with the device

    core::cuda::DeviceStruct<Parameters> dev_pars;

    void sync_pars() {
        if (!pars_synced)
            dev_pars.upload(pars);
    }

  public:
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

// make maps as CudaArray2D
#ifdef TEMPLATE_CLASS_MAPS
#define X(TYPE, NAME, DESCRIPTION) \
    core::cuda::CudaArray2D<TYPE> NAME;
    TEMPLATE_CLASS_MAPS
#undef X
#endif

// private device arrays
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    core::cuda::DeviceArray<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    // make enumerators
#ifdef TEMPLATE_CLASS_TYPES
    enum class Type {
#define X(NAME, DESCRIPTION) \
    NAME,
        TEMPLATE_CLASS_TYPES
#undef X
    };
#endif

// get all the map pointers in a structure (not yet used generally)
#ifdef TEMPLATE_CLASS_MAPS
    MapPointers get_map_pointers() {
        MapPointers result;
#define X(TYPE, NAME, DESCRIPTION) \
    result.NAME = NAME.dev_ptr();
        TEMPLATE_CLASS_MAPS
#undef X
        return result;
    }
#endif

// NEW DeviceArray2D hooks
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION) \
    core::cuda::DeviceArray2D<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif

    void allocate_device();
    void deallocate_device();

    void process();

    core::cuda::DeviceArray2D<float> device_array_2d;
};

} // namespace TEMPLATE_NAMESPACE
