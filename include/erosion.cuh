/*

REFACTOR FROM TINY ERODE, WORKING BUT DIAGONALLY

we are in this conversation:

https://copilot.microsoft.com/chats/pcsfGy53kozEuavmq68Fu

*/
#pragma once

// #define ENABLE_EROSION_TILED_MEMORY // 🚧 memory optimization, currently we just have global memory of entire map

// #define EROSION_BLOCK_SIZE 16 // normally the best size for block, 16x16 = 256 threads per block 8 warps (32 threads a warp)
// #define EROSION_BLOCK_SIZE 8 // smaller blocks (not a good idea)

#include "core.h"
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>

#define EROSION_PARAMETERS           \
    X(float, min_height, 0.0)        \
    X(float, max_height, 1.0)        \
    X(float, jitter, 0.0)            \
    X(float, rain_rate, 0.0)         \
    X(float, evaporation_rate, 0.01) \
    X(float, erosion_rate, 0.01)     \
    X(float, deposition_rate, 0.01)  \
    X(float, slope_threshold, 0.01)  \
    X(float, flow_factor, 0.1)       \
    X(int, steps, 1024)              \
    X(int, block_size, 16)           \
    X(int, mode, 0)                  \
    X(bool, wrap, true)

#define EROSION_MAPS      \
    X(float, height) \
    X(float, water)  \
    X(float, sediment)

#define CUDA_CHECK(call)                                                                               \
    do {                                                                                               \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

namespace erosion {

struct Parameters {
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE NAME = DEFAULT_VAL;
    EROSION_PARAMETERS
#undef X
};

class Erosion {

  private:
    // curandState *dev_rand_states = nullptr;

    Parameters host_pars;
    Parameters *dev_pars = nullptr;

    // device maps
#define X(TYPE, NAME) \
    TYPE *dev_##NAME##_map = nullptr;
    EROSION_MAPS
#undef X



    size_t width = 0; // 0 until
    size_t height = 0;

  public:
#define X(TYPE, NAME, DEFAULT_VAL)               \
    TYPE get_##NAME() { return host_pars.NAME; } \
    void set_##NAME(TYPE value) { host_pars.NAME = value; }
    EROSION_PARAMETERS
#undef X

    // get/set height and width
    size_t get_width() { return width; }
    size_t get_height() { return height; }
    void set_width(size_t value) { width = value; }
    void set_height(size_t value) { height = value; }

    void allocate_host_memory();
    void free_host_memory();

    void allocate_device_memory();
    void copy_to_device();
    void copy_from_device();
    void free_device_memory();

    void run_erosion(float *host_data);



  // HOST MEMORY

  // host maps as std::vector
#define X(TYPE, NAME) \
    std::vector<TYPE> host_##NAME##_map;
    EROSION_MAPS
#undef X





};

inline void Erosion::allocate_host_memory() {
    free_host_memory();
}

inline void Erosion::free_host_memory() {
}

inline void Erosion::allocate_device_memory() {

    free_device_memory(); // free memory if already allocated

    // copy allocate and copy pars to gpu
    CUDA_CHECK(cudaMalloc(&dev_pars, sizeof(Parameters)));
    CUDA_CHECK(cudaMemcpy(dev_pars, &host_pars, sizeof(Parameters), cudaMemcpyHostToDevice));

    size_t size = width * height;

    // alocate maps
#define X(TYPE, NAME) \
    CUDA_CHECK(cudaMalloc(&dev_##NAME##_map, size * sizeof(TYPE)));
    EROSION_MAPS
#undef X
}

inline void Erosion::copy_to_device() {
}

inline void Erosion::copy_from_device() {
}

inline void Erosion::free_device_memory() {

    // free pars
    if (dev_pars) {
        CUDA_CHECK(cudaFree(dev_pars));
    }

    // free maps
#define X(TYPE, NAME)                           \
    if (dev_##NAME##_map) {                     \
        CUDA_CHECK(cudaFree(dev_##NAME##_map)); \
    }
    EROSION_MAPS
#undef X
}

} // namespace erosion