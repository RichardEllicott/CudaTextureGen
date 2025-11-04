/*

🧜‍♀️ TEMPLATE VERSION 20251027-3

this one uses more clever types and classes, needing less horrible macros

*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME FluidSimulation
#define TEMPLATE_NAMESPACE fluid_simulation

#define TEMPLATE_CLASS_PARAMETERS \
    X(size_t, width, 1024)        \
    X(size_t, height, 1024)       \
    X(size_t, _block, 16)         \
    X(float, gravity, 9.8)        \
    X(float, dt, 1.0)             \
    X(float, cell_size, 1.0)      \
    X(float, steps, 16)

#define TEMPLATE_CLASS_MAPS  \
    X(float, height_map)     \
    X(float, height_map_out) \
    X(float, velocity_map)   \
    X(float, velocity_map_out)

#define TEMPLATE_CLASS_TYPES \
    X(APPLE)                 \
    X(ORANGE)                \
    X(POTATO)

// ════════════════════════════════════════════════ //

#include "cuda_types.cuh"

namespace TEMPLATE_NAMESPACE {

struct Parameters {
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif
};

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

      private:
        core::cuda::Struct<Parameters> dev_pars;

      public:
        void allocate_device();
        void deallocate_device();

    void process();
};

} // namespace TEMPLATE_NAMESPACE
