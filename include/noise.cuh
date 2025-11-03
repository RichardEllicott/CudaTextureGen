/*

🧜‍♀️ TEMPLATE VERSION 20251027-3
🧜‍♀️ TEMPLATE VERSION 20251102-1 // added a structure for map pointers

this one uses more clever types and classes, needing less horrible macros


⚠️⚠️⚠️ THIS ONE DOESN'T SUPPORT ANY SORTS OF RGB NUMPY ARRAYS ETC ⚠️⚠️⚠️



🎃 this noise generator was trying to work on 3D, also investigating animation and periodicity


*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME Noise
#define TEMPLATE_NAMESPACE noise

#define TEMPLATE_CLASS_PARAMETERS \
    X(size_t, width, 1024)        \
    X(size_t, height, 1024)       \
    X(size_t, _block, 16)         \
    X(int, type, 0)               \
    X(int, seed, 0)               \
    X(float, period_x, 7.0f)      \
    X(float, period_y, 7.0f)      \
    X(float, period_z, 7.0f)      \
    X(float, x, 0.0f)             \
    X(float, y, 0.0f)             \
    X(float, z, 0.0f)             \
    X(float, warp_amp, 4.0f)      \
    X(float, warp_scale, 1.0f)    \
    X(float, angle, 0.0f)

#define TEMPLATE_CLASS_MAPS \
    X(float, image)

// // trying to support float3
#define TEMPLATE_CLASS_MAPS2 \
    X(float3, image_rgb)

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


// custom set all period's at once
    float get_period() {
        return get_period_x();
    };
    void set_period(float value) {
        set_period_x(value);
        set_period_y(value);
        set_period_z(value);
    };

    void process();
};

} // namespace TEMPLATE_NAMESPACE
