/*

🧜‍♀️ TEMPLATE VERSION 20251027-3

this one uses more clever types and classes, needing less horrible macros

*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME Erosion3
#define TEMPLATE_NAMESPACE erosion3


#define TEMPLATE_CLASS_PARAMETERS    \
    X(size_t, _block, 16)            \
    X(size_t, width, 256)            \
    X(size_t, height, 256)           \
    X(int, steps, 1024)              \
    X(float, rain_rate, 0.01)        \
    X(bool, wrap, true)              \
    X(float, max_water_outflow, 1.0) \
    X(float, capacity, 0.1)          \
    X(float, erode, 0.1)             \
    X(float, deposit, 0.1)           \
    X(float, evaporation_rate, 0.1)  \
    X(bool, debug_hash_cell_order, false)



#define TEMPLATE_CLASS_MAPS \
    X(float, height_map)    \
    X(float, water_map)     \
    X(float, sediment_map)  \
    X(float, dh_out)        \
    X(float, ds_out)        \
    X(float, dw_out)



    // private device arrays
#define TEMPLATE_CLASS_DEVICE_ARRAYS \
    X(float, height_map_out)         \
    X(float, water_map_out)          \
    X(float, sediment_map_out)       \
    X(float, flux8)                  \




// ════════════════════════════════════════════════ //

#include "cuda_types.cuh"

namespace TEMPLATE_NAMESPACE {

// struct FluxCell {
//     float f[8];
// };

struct Parameters {
    // declare pars on structures
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
};

class TEMPLATE_CLASS_NAME {

  private:
    Parameters pars;
    bool device_allocated = false;





    size_t _count = 0; // count of passes

  public:
    // make getter/setters for the pars
#define X(TYPE, NAME, DEFAULT_VAL)                \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
    TEMPLATE_CLASS_PARAMETERS
#undef X

    // make maps
#define X(TYPE, NAME) \
    core::cuda::CudaArray2D<TYPE> NAME;
    TEMPLATE_CLASS_MAPS
#undef X

    // private device arrays
#define X(TYPE, NAME) \
    core::cuda::DeviceArray<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X



    void allocate_device(); // and upload
    // void upload_device();
    void deallocate_device(); // and download
    // void download_device();

    void process();

    TEMPLATE_CLASS_NAME();
    ~TEMPLATE_CLASS_NAME();
};

} // namespace TEMPLATE_NAMESPACE
