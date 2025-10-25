/*

new Erosion2

this one uses my TemplateClass

it should be an identical copy (in the header and bind files), of TemplateClass
note we change the defines shown in between the 📜 symbols




⚠️⚠️⚠️⚠️⚠️⚠️⚠️
DID NOT WORK!!!
⚠️⚠️⚠️⚠️⚠️⚠️⚠️


*/
#pragma once

#include "core.h"
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

// 📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜

#define TEMPLATE_CLASS_NAME Erosion2
#define TEMPLATE_CLASS_NAMESPACE erosion2

#define TEMPLATE_CLASS_PARAMETERS      \
    X(size_t, width, 256)              \
    X(size_t, height, 256)             \
    X(float, rain_rate, 0.01f)         \
    X(float, evaporation_rate, 0.005f) \
    X(float, erosion_rate, 0.01f)      \
    X(float, deposition_rate, 0.25f)   \
    X(float, slope_threshold, 0.1f)    \
    X(int, steps, 128)

#define TEMPLATE_CLASS_MAPS \
    X(float, height_map)    \
    X(float, water_map)     \
    X(float, sediment_map)

// 📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜

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

// The template class itself
class TEMPLATE_CLASS_NAME {

  private:
    Parameters host_pars; // pars stored host side
    Parameters *device_pars = nullptr;

    Maps host_maps;
    Maps device_map_pointers;          // host-side struct holding device pointers
    Maps *device_map_struct = nullptr; // device-side struct to pass to kernels

  public:
    TEMPLATE_CLASS_NAME() {
        // set default pars
#define X(TYPE, NAME, DEFAULT_VAL) \
    host_pars.NAME = DEFAULT_VAL;
        TEMPLATE_CLASS_PARAMETERS
#undef X

        // null the maps
#define X(TYPE, NAME) \
    host_maps.NAME = nullptr;
        TEMPLATE_CLASS_MAPS
#undef X
    }

    ~TEMPLATE_CLASS_NAME() {
        free_memory();
    }

// Decalare Get/Sets
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE get_##NAME() const;       \
    void set_##NAME(TYPE value);
    TEMPLATE_CLASS_PARAMETERS
#undef X

// Decalare Get/Sets for maps
#define X(TYPE, NAME)         \
    TYPE *get_##NAME() const; \
    void set_##NAME(TYPE *ptr);
    TEMPLATE_CLASS_MAPS
#undef X

    //  Allocate Memory
    void allocate_and_copy_to_gpu() {

        // free_memory(); // Free existing allocations first

        size_t map_size = host_pars.width * host_pars.height; // find map size

        // allocate and copy map data to GPU if we have any (we can leave some maps null)
#define X(TYPE, NAME)                                                                                                      \
    if (host_maps.NAME) {                                                                                                  \
        CUDA_CHECK(cudaMalloc(&device_map_pointers.NAME, map_size * sizeof(TYPE)));                                        \
        CUDA_CHECK(cudaMemcpy(device_map_pointers.NAME, host_maps.NAME, map_size * sizeof(TYPE), cudaMemcpyHostToDevice)); \
    } else {                                                                                                               \
        device_map_pointers.NAME = nullptr;                                                                                \
    }
        TEMPLATE_CLASS_MAPS
#undef X

        // copy pars to gpu
        CUDA_CHECK(cudaMalloc(&device_pars, sizeof(Parameters)));
        CUDA_CHECK(cudaMemcpy(device_pars, &host_pars, sizeof(Parameters), cudaMemcpyHostToDevice));

        // copy map pointers to gpu
        CUDA_CHECK(cudaMalloc(&device_map_struct, sizeof(Maps)));
        CUDA_CHECK(cudaMemcpy(device_map_struct, &device_map_pointers, sizeof(Maps), cudaMemcpyHostToDevice));
    }

    // copy data back to the maps
    void copy_maps_back_from_gpu() {

        size_t map_size = host_pars.width * host_pars.height; // find map size

#define X(TYPE, NAME)                                                                                                      \
    if (host_maps.NAME && device_map_pointers.NAME) {                                                                      \
        CUDA_CHECK(cudaMemcpy(host_maps.NAME, device_map_pointers.NAME, map_size * sizeof(TYPE), cudaMemcpyDeviceToHost)); \
    }
        TEMPLATE_CLASS_MAPS
#undef X
    }

    // Free Memory
    void free_memory() {

#define X(TYPE, NAME)                       \
    if (device_map_pointers.NAME) {         \
        cudaFree(device_map_pointers.NAME); \
        device_map_pointers.NAME = nullptr; \
    }
        TEMPLATE_CLASS_MAPS
#undef X

        if (device_pars) {
            cudaFree(device_pars);
            device_pars = nullptr;
        }

        if (device_map_struct) {
            cudaFree(device_map_struct);
            device_map_struct = nullptr;
        }
    }

    // 🚀
    void process();
};

// get/set pars
#define X(TYPE, NAME, DEFAULT_VAL)                                                 \
    inline TYPE TEMPLATE_CLASS_NAME::get_##NAME() const { return host_pars.NAME; } \
    inline void TEMPLATE_CLASS_NAME::set_##NAME(TYPE value) { host_pars.NAME = value; }
TEMPLATE_CLASS_PARAMETERS
#undef X

// get/set maps
#define X(TYPE, NAME)                                                               \
    inline TYPE *TEMPLATE_CLASS_NAME::get_##NAME() const { return host_maps.NAME; } \
    inline void TEMPLATE_CLASS_NAME::set_##NAME(TYPE *ptr) { host_maps.NAME = ptr; }
TEMPLATE_CLASS_MAPS
#undef X

} // namespace TEMPLATE_CLASS_NAMESPACE

// 📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜

#undef TEMPLATE_CLASS_NAME
#undef TEMPLATE_CLASS_NAMESPACE
#undef TEMPLATE_CLASS_PARAMETERS
#undef TEMPLATE_CLASS_MAPS
