/*

🧜‍♀️ TEMPLATE VERSION 20251109-1

features:

auto paramaters (easiest way of sending lots of pars)
auto maps (floats only)
auto device only arrays (support different data types and multidimensions)
enumerators
debug data container
map pointer structure (UNUSED, i prefer to hardcode map pars allowing extra const and __restrict__)


*/
#pragma once

// ================================================================ //
#define TEMPLATE_CLASS_NAME TemplateClass4
#define TEMPLATE_NAMESPACE template_class_4

// auto set up pars (added to python and to pars object for upload)
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                             \
    X(size_t, _width, 1024, "map width")                      \
    X(size_t, _height, 1024, "map height")                    \
    X(size_t, _block, 16, "block size (best to leave at 16)") \
    X(bool, test_bool, 0.0, "test bool")                      \
    X(float, test_float, 0.0, "test float")                   \
    X(int, test_int, 0.0, "test int")

// auto set up maps (have local copy and upload to gpu) ⚠️ only supports floats!
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_MAPS          \
    X(float, image, "example image") \
    X(float, height_map, "example height_map")

// private arrays, these can be multi-dimensional and are GPU side only
// (TYPE, DIMENSION, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAYS                        \
    X(float, 1, height_map_out, "second height buffer")     \
    X(float, 1, water_map_out, "second water buffer")       \
    X(float, 1, sediment_map_out, "second sediment buffer") \
    X(float, 8, flux8, "water flow out to 8 neighbours")

// enumerators
// (NAME, DESCRIPTION)
#define TEMPLATE_CLASS_TYPES        \
    X(APPLE, "apple description")   \
    X(ORANGE, "orange description") \
    X(POTATO, "potato description")

// debug data container, used to collect data for debug purposes
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_DEBUG_DATA   \
    X(float, total_height, 0.0, "") \
    X(float, total_water, 0.0, "")  \
    X(float, total_sediment, 0.0, "")

// DeviceArray2D
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAY_2DS \
    X(float, device_array_2d_test, "testing device array")




// ================================================================ //

#include "core/cuda/types.cuh"

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
#ifdef TEMPLATE_CLASS_MAPS
struct MapPointers {
#define X(TYPE, NAME, DESCRIPTION) \
    TYPE *NAME = nullptr;
    TEMPLATE_CLASS_MAPS
#undef X
};
#endif

class TEMPLATE_CLASS_NAME {

// make getter/setters for the pars
#ifdef TEMPLATE_CLASS_PARAMETERS
  private:
    Parameters pars;

  public:
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION)   \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
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
    core::cuda::DeviceArray1D<TYPE> NAME;
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

// NEW DeviceArray2D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION) \
    core::cuda::DeviceArray2D<TYPE> NAME;
    TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif

    void process();
};

} // namespace TEMPLATE_NAMESPACE
