/*

simple erosion that distributes sediment to neighbours

🧜‍♀️ TEMPLATE VERSION 20251102-1

*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME Erosion4
#define TEMPLATE_NAMESPACE erosion4

#define TEMPLATE_CLASS_PARAMETERS   \
    X(size_t, _width, 1024)          \
    X(size_t, _height, 1024)         \
    X(size_t, _block, 16)           \
    X(int, steps, 512)              \
    X(bool, wrap, true)             \
    X(float, jitter, 0.0)           \
    X(float, erosion_rate, 0.0)    \
    X(float, slope_threshold, 0.0) \
    X(float, deposition_rate, 0.0) \
    X(int, mode, 0)


    // good settings
// heightmap_scale = 16.0 * 4
// erosion.erosion_rate = 0.01
// erosion.deposition_rate = 0.01
// erosion.slope_threshold = 1.0


#define TEMPLATE_CLASS_MAPS \
    X(float, height_map)    \
    X(float, sediment_map)

// // trying to support float3
// #define TEMPLATE_CLASS_MAPS2 \
//     X(float3, image)

// #define TEMPLATE_CLASS_TYPES \
//     X(APPLE)                 \
//     X(ORANGE)                \
//     X(POTATO)

// ════════════════════════════════════════════════ //

#include "core/cuda/types.cuh"
#include "core/cuda/cuda_array_2d.cuh"

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

    void process();
};

} // namespace TEMPLATE_NAMESPACE
