/*

should work as new template

has rule of 5 issue?? not likely a problem just using with python binds though

https://claude.ai/chat/ed047b28-6b69-4ef8-97f9-993b7351a5e4


*/
#pragma once

#include "core.h"
#include <cstdio>

// 📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜

#define TEMPLATE_CLASS_NAME TemplateClass2
#define TEMPLATE_CLASS_NAMESPACE template_class2

#define TEMPLATE_CLASS_PARAMETERS \
    X(size_t, width, 256)         \
    X(size_t, height, 256)        \
    X(size_t, _block, 16)          \
    X(float, test_par1, 0.0)      \
    X(float, test_par2, 1.0)      \
    X(float, test_par3, 1.0)

#define TEMPLATE_CLASS_MAPS \
    X(float, height_map)    \
    X(float, blend_mask)    \
    X(float, gradient_map)

// 📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜

// calling cuda functions checking for errors
#define CUDA_CHECK(call)                                                                               \
    do {                                                                                               \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

namespace TEMPLATE_CLASS_NAMESPACE {

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

class TEMPLATE_CLASS_NAME {
  private:
    Parameters pars;
    Parameters *dev_pars = nullptr;

    Maps maps;                // prt's to maps on device
    Maps *dev_maps = nullptr; // ptr to Maps struct on device

  public:
    // get/set pars
#define X(TYPE, NAME, DEFAULT_VAL)                                     \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
    TEMPLATE_CLASS_PARAMETERS
#undef X

// host maps as std::vector
#define X(TYPE, NAME) \
    core::Array2D<TYPE> NAME;
    TEMPLATE_CLASS_MAPS
#undef X

    void allocate_device_memory() {

        free_device_memory();

        // copy allocate and copy pars to gpu
        CUDA_CHECK(cudaMalloc(&dev_pars, sizeof(Parameters)));
        CUDA_CHECK(cudaMemcpy(dev_pars, &pars, sizeof(Parameters), cudaMemcpyHostToDevice));

        size_t size = pars.width * pars.height;

        // alocate maps
#define X(TYPE, NAME) \
    CUDA_CHECK(cudaMalloc(&maps.NAME, size * sizeof(TYPE)));
        TEMPLATE_CLASS_MAPS
#undef X

        // send the map ptrs after allocating the maps
        CUDA_CHECK(cudaMalloc(&dev_maps, sizeof(Maps)));
        CUDA_CHECK(cudaMemcpy(dev_maps, &maps, sizeof(Maps), cudaMemcpyHostToDevice));
    }
    void copy_maps_to_device() {

        size_t size = pars.width * pars.height; // ⚠️ WARNING if the size of the map changes here, we could have issues

        // if (maps.height_map && height_map.get_width() == pars.width && height_map.get_height() == pars.height) {
        //     CUDA_CHECK(cudaMemcpy(maps.height_map, height_map.data(), size * sizeof(float), cudaMemcpyHostToDevice)); // copy data
        // }

#define X(TYPE, NAME)                                                                                \
    if (maps.NAME && NAME.get_width() == pars.width && NAME.get_height() == pars.height) {           \
        CUDA_CHECK(cudaMemcpy(maps.NAME, NAME.data(), size * sizeof(TYPE), cudaMemcpyHostToDevice)); \
    }
        TEMPLATE_CLASS_MAPS
#undef X
        //
        //
    }

    void copy_maps_from_device() {

        size_t size = pars.width * pars.height; // ⚠️ WARNING if the size of the map changes here, we could have issues

#define X(TYPE, NAME)                                                                                \
    if (maps.NAME) {                                                                                 \
        NAME.resize(pars.width, pars.height);                                                        \
        CUDA_CHECK(cudaMemcpy(NAME.data(), maps.NAME, size * sizeof(TYPE), cudaMemcpyDeviceToHost)); \
    }
        TEMPLATE_CLASS_MAPS
#undef X
        //
        //
    }
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
        //
        //
        //
    }

    void process();

    TEMPLATE_CLASS_NAME() {

        // set default vals
#define X(TYPE, NAME, DEFAULT_VAL) \
    pars.NAME = DEFAULT_VAL;
        TEMPLATE_CLASS_PARAMETERS
#undef X
    }

    ~TEMPLATE_CLASS_NAME() {

        free_device_memory();
    }
};





} // namespace TEMPLATE_CLASS_NAMESPACE

// #undef TEMPLATE_CLASS_NAME
// #undef TEMPLATE_CLASS_NAMESPACE
// #undef TEMPLATE_CLASS_PARAMETERS
// #undef TEMPLATE_CLASS_MAPS
