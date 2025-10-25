/*

REFACTOR FROM TINY ERODE, WORKING BUT DIAGONALLY

we are in this conversation:

https://copilot.microsoft.com/chats/pcsfGy53kozEuavmq68Fu

*/
#pragma once

// #define ENABLE_EROSION_JITTER // adds jitter to erosion
// #define ENABLE_EROSION_TILED_MEMORY // 🚧 memory optimization, currently we just have global memory of entire map

#define EROSION_BLOCK_SIZE 16 // normally the best size for block, 16x16 = 256 threads per block 8 warps (32 threads a warp)
// #define EROSION_BLOCK_SIZE 8 // smaller blocks (not a good idea)

#include "core.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#ifdef ENABLE_EROSION_JITTER
#include <curand_kernel.h>
#endif

// constants to be copied to the GPU
#define EROSION_PARAMETERS          \
    X(float, min_height, 0.0)       \
    X(float, max_height, 1.0)       \
    X(float, jitter, 0.0)           \
    X(float, rain_rate, 0.0)        \
    X(float, evaporation_rate, 0.0) \
    X(float, erosion_rate, 0.01)    \
    X(float, deposition_rate, 0.01) \
    X(float, slope_threshold, 0.01) \
    X(int, steps, 1024)             \
    X(bool, wrap, true)

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

    Parameters pars;
    Parameters *dev_pars;

    float *dev_heightmap = nullptr;
    float *dev_water = nullptr;
    // float *dev_outflow = nullptr;
    float *dev_sediment = nullptr;

  public:
#define X(TYPE, NAME, DEFAULT_VAL)               \
    TYPE get_##NAME() { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
    EROSION_PARAMETERS
#undef X

    void run_erosion(float *host_data, int width, int height);

    Erosion();
    ~Erosion();
};

} // namespace erosion