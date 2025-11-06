/*

🧜‍♀️ TEMPLATE VERSION 20251102-1


this is a copy of Erosion3, which works so this is fine to mess around with

*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME Erosion5
#define TEMPLATE_NAMESPACE erosion5

#define TEMPLATE_CLASS_PARAMETERS    \
    X(size_t, _block, 16)            \
    X(size_t, width, 256)            \
    X(size_t, height, 256)           \
    X(int, mode, 0)                  \
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

// // trying to support float3
// #define TEMPLATE_CLASS_MAPS2 \
//     X(float3, image)

// #define TEMPLATE_CLASS_TYPES \
//     X(APPLE)                 \
//     X(ORANGE)                \
//     X(POTATO)

// ════════════════════════════════════════════════ //

#include "cuda_types.cuh"
// #include <std>

namespace TEMPLATE_NAMESPACE {

// set to kernels as pointer
struct Parameters {
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif
};
// OPTIONAL Compile‑time safety check
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy");

// could be used with kernels
struct MapPointers {
#ifdef TEMPLATE_CLASS_MAPS
#define X(TYPE, NAME) \
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

  public:
#define X(TYPE, NAME, DEFAULT_VAL)                \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

// make maps as CudaArray2D
#ifdef TEMPLATE_CLASS_MAPS
#define X(TYPE, NAME) \
    core::cuda::CudaArray2D<TYPE> NAME;
    TEMPLATE_CLASS_MAPS
#undef X
#endif

    // make enumerators
#ifdef TEMPLATE_CLASS_TYPES
    enum class Type {
#define X(NAME) \
    NAME,
        TEMPLATE_CLASS_TYPES
#undef X
    };
#endif

// get all the map pointers in a structure
#ifdef TEMPLATE_CLASS_MAPS
    MapPointers get_map_pointers() {
        MapPointers result;

#define X(TYPE, NAME) result.NAME = NAME.dev_ptr();
        TEMPLATE_CLASS_MAPS
#undef X

        return result;
    }
#endif

    //
    //
    //
    //
  private:
    bool device_allocated = false;

    // pointers for swapping the maps around (ping/pong)
    float *h_cur = nullptr;
    float *w_cur = nullptr;
    float *s_cur = nullptr;
    float *h_next = nullptr;
    float *w_next = nullptr;
    float *s_next = nullptr;

    core::cuda::DeviceArray<float> flux8; // details movement to the adjacent cells
    core::cuda::DeviceArray<float> height_map_out;
    core::cuda::DeviceArray<float> water_map_out;
    core::cuda::DeviceArray<float> sediment_map_out;

    size_t _count = 0; // count of passes

  public:
    void process();

    void allocate_device(); // and upload
    // void upload_device();
    void deallocate_device(); // and download
    // void download_device();

    TEMPLATE_CLASS_NAME();
    ~TEMPLATE_CLASS_NAME();
};

} // namespace TEMPLATE_NAMESPACE
