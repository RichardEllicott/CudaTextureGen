/*

🧜‍♀️ TEMPLATE VERSION 20251102-1 // added a structure for map pointers

this one uses more clever types and classes, needing less horrible macros


⚠️⚠️⚠️ THIS ONE DOESN'T SUPPORT ANY SORTS OF RGB NUMPY ARRAYS ETC ⚠️⚠️⚠️

*/
#pragma once

// ================================================================
#define TEMPLATE_CLASS_NAME TemplateClass3
#define TEMPLATE_NAMESPACE template_class_3

// paramaters
#define TEMPLATE_CLASS_PARAMETERS \
    X(size_t, width, 1024)        \
    X(size_t, height, 1024)       \
    X(size_t, _block, 16)         \
    X(float, test, 0.0)

// 2D arrays with local copies
#define TEMPLATE_CLASS_MAPS \
    X(float, image)


// enumerators
#define TEMPLATE_CLASS_TYPES \
    X(APPLE)                 \
    X(ORANGE)                \
    X(POTATO)

// ================================================================

#include "cuda_types.cuh"

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

    // testing new object:

    core::cuda::CudaArray3D<float> array3d_test;

    void process();
};

} // namespace TEMPLATE_NAMESPACE
