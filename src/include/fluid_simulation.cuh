/*

🧜‍♀️ TEMPLATE VERSION 20251109-1

this one uses more clever types and classes, needing less horrible macros

*/
#pragma once

// ================================================================ //
#define TEMPLATE_CLASS_NAME FluidSimulation
#define TEMPLATE_NAMESPACE fluid_simulation

// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS         \
    X(size_t, width, 1024, "")            \
    X(size_t, height, 1024, "")           \
    X(size_t, _block, 16, "")             \
    X(float, gravity, 9.8, "")            \
    X(float, dt, 1.0, "")                 \
    X(float, cell_size, 1.0, "")          \
    X(float, steps, 1, "")                \
    X(float, wave_speed, 1.0, "")         \
    X(float, damping, 0.0, "mode 1 only") \
    X(int, mode, 1, "")\
    X(bool, wrap, true, "")

// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_2DS                                      \
    X(float, water_map, "the main map to set")                               \
    X(float, water_map_previous, "must start with a copy of water_map")      \
    X(float, water_map_next, "weill be written to, can start uninitialized") \
    X(float, height_map, "testing device array")

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

class TEMPLATE_CLASS_NAME {


  private:
    Parameters pars;
    core::cuda::DeviceStruct<Parameters> dev_pars;

    bool pars_synced = false; // record if the pars have been synced since update
    void sync_pars() {
        if (!pars_synced) {
            dev_pars.upload(pars);
            pars_synced = true;
        }
    }

    core::cuda::Stream stream; // will be allocated along with object

  public:

// make getter/setters for the pars
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
