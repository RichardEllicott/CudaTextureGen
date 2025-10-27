/*

should work as new template

TRYING TO ADD NEW SAFTEY


https://claude.ai/chat/1eb7983b-ca1f-436c-b1eb-75fcb28f319f


*/
#pragma once

#include "core.h"
#include <cstdio>

#pragma region XMACRO_SETTINGS
// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME TemplateClass2
#define TEMPLATE_CLASS_NAMESPACE template_class2

#define TEMPLATE_CLASS_PARAMETERS \
    X(size_t, width, 256)         \
    X(size_t, height, 256)        \
    X(size_t, _block, 16)         \
    X(float, test, 0.0)

#define TEMPLATE_CLASS_MAPS \
    X(float, height_map)    \
    X(float, blend_mask)
// ════════════════════════════════════════════════ //
#pragma endregion

#pragma region MACROS

// calling cuda functions checking for errors
#define CUDA_CHECK(call)                                                                               \
    do {                                                                                               \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

#pragma endregion

namespace TEMPLATE_CLASS_NAMESPACE {

#pragma region STRUCTURES

struct Parameters {
    // declare pars on structures
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE NAME;
    TEMPLATE_CLASS_PARAMETERS
#undef X
};

struct Maps {
    // declare maps on structures
#define X(TYPE, NAME) \
    TYPE *NAME = nullptr;
    TEMPLATE_CLASS_MAPS
#undef X
};

#pragma endregion

class TEMPLATE_CLASS_NAME {

#pragma region PARS

  private:
    Parameters pars;                // host side pars
    Parameters *dev_pars = nullptr; // pointer to device side pars

    Maps maps;                // pointer to device side maps
    Maps *dev_maps = nullptr; // pointer to device side structure containing map pointers

  public:
    // make getter/setters for the pars
#define X(TYPE, NAME, DEFAULT_VAL)                \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
    TEMPLATE_CLASS_PARAMETERS
#undef X

// host side maps as core::Array2D ... these can automaticly be copied to and from the device
#define X(TYPE, NAME) \
    core::Array2D<TYPE> NAME;
    TEMPLATE_CLASS_MAPS
#undef X

#pragma endregion

#pragma region MEMORY_MANAGEMENT_FUNCTIONS

  private:
    size_t allocated_array_size = 0;

  public:
    // allocate device memory, also copy accross pars
    void allocate_device_memory() {

        free_device_memory();

        // copy allocate and copy pars to gpu
        CUDA_CHECK(cudaMalloc(&dev_pars, sizeof(Parameters)));
        CUDA_CHECK(cudaMemcpy(dev_pars, &pars, sizeof(Parameters), cudaMemcpyHostToDevice)); // ⚠️ we copy pars here (confusing?)

        size_t array_size = pars.width * pars.height;

        allocated_array_size = array_size;

        // alocate maps
#define X(TYPE, NAME) \
    CUDA_CHECK(cudaMalloc(&maps.NAME, array_size * sizeof(TYPE)));
        TEMPLATE_CLASS_MAPS
#undef X

        // send the map ptrs after allocating the maps
        CUDA_CHECK(cudaMalloc(&dev_maps, sizeof(Maps)));
        CUDA_CHECK(cudaMemcpy(dev_maps, &maps, sizeof(Maps), cudaMemcpyHostToDevice));
    }

    // set all the maps on device to 0 (clearing the memory)
    void clear_maps_on_device() {

        size_t array_size = pars.width * pars.height;

        // if the size has changed, we will trigger a reallocation of memory
        if (array_size != allocated_array_size) {
            allocate_device_memory();
        }

        // set all allocated maps to 0
#define X(TYPE, NAME)                                                    \
    if (maps.NAME) {                                                     \
        CUDA_CHECK(cudaMemset(maps.NAME, 0, array_size * sizeof(TYPE))); \
    };
        TEMPLATE_CLASS_MAPS
#undef X
    }

    // copy over the map data from the local core::Array2D's to the device
    void copy_maps_to_device() {

        size_t array_size = pars.width * pars.height;

        // if the size has changed, we will trigger a reallocation of memory
        if (array_size != allocated_array_size) {
            allocate_device_memory();
        }

        // copy maps accross, only copy ones where the width and height matches though
#define X(TYPE, NAME)                                                                                      \
    if (maps.NAME && NAME.get_width() == pars.width && NAME.get_height() == pars.height) {                 \
        CUDA_CHECK(cudaMemcpy(maps.NAME, NAME.data(), array_size * sizeof(TYPE), cudaMemcpyHostToDevice)); \
    }
        TEMPLATE_CLASS_MAPS
#undef X
        //
        //
    }

    // copy back the maps from the device
    void copy_maps_from_device() {

        size_t array_size = pars.width * pars.height; // ⚠️ WARNING if the size of the map changes here, we could have issues

#define X(TYPE, NAME)                                                                                      \
    if (maps.NAME) {                                                                                       \
        NAME.resize(pars.width, pars.height);                                                              \
        CUDA_CHECK(cudaMemcpy(NAME.data(), maps.NAME, array_size * sizeof(TYPE), cudaMemcpyDeviceToHost)); \
    }
        TEMPLATE_CLASS_MAPS
#undef X
        //
        //
    }

    // free the device memory
    void free_device_memory() {

        // free pars on gpu
        if (dev_pars) {
            CUDA_CHECK(cudaFree(dev_pars));
            dev_pars = nullptr;
        }

        // free maps struct on gpu
        if (dev_maps) {
            CUDA_CHECK(cudaFree(dev_maps));
            dev_maps = nullptr;
        }

        // free maps memory on gpu
#define X(TYPE, NAME)                    \
    if (maps.NAME) {                     \
        CUDA_CHECK(cudaFree(maps.NAME)); \
        maps.NAME = nullptr;             \
    }
        TEMPLATE_CLASS_MAPS
#undef X

        allocated_array_size = 0;
    }

#pragma endregion

#pragma region CONSTRUCTORS
    // constructor, set the default values
    TEMPLATE_CLASS_NAME() {
        // set the default values
#define X(TYPE, NAME, DEFAULT_VAL) \
    pars.NAME = DEFAULT_VAL;
        TEMPLATE_CLASS_PARAMETERS
#undef X
    }

    // deconstructor, ensure device memory freed
    ~TEMPLATE_CLASS_NAME() {
        free_device_memory();
    }

    // Rule of Five, prevent copying or implement proper deep copy
    TEMPLATE_CLASS_NAME(const TEMPLATE_CLASS_NAME &) = delete;
    TEMPLATE_CLASS_NAME &operator=(const TEMPLATE_CLASS_NAME &) = delete;
    TEMPLATE_CLASS_NAME(TEMPLATE_CLASS_NAME &&) = delete;
    TEMPLATE_CLASS_NAME &operator=(TEMPLATE_CLASS_NAME &&) = delete;

#pragma endregion

    // run default process
    void process();
};

} // namespace TEMPLATE_CLASS_NAMESPACE
