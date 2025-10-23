/*

REFACTOR FROM TINY ERODE, WORKING BUT DIAGONALLY

we are in this conversation:

https://copilot.microsoft.com/chats/pcsfGy53kozEuavmq68Fu

*/
#pragma once

#define ENABLE_EROSION_JITTER // adds jitter to erosion
// #define ENABLE_EROSION_TILED_MEMORY // 🚧 memory optimization, currently we just have global memory of entire map
#define ENABLE_EROSION_WRAP     // erosion wraps from the edges, making the result tileable
// #define ENABLE_EROSION_TRIPWIRE // tripwire error to prevent multiple instances of this class

#include "core.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#ifdef ENABLE_EROSION_JITTER
#include <curand_kernel.h>
#endif

// constants to be copied to the GPU
#define EROSION_CONSTANTS     \
    X(float, min_height, 0.0) \
    X(float, max_height, 1.0) \
    X(float, jitter, 1.0)

namespace erosion {

// --------------------------------------------------------------------------------
// Declare CUDA constants
#define X(TYPE, NAME, DEFAULT_VAL) \
    __constant__ TYPE NAME;
EROSION_CONSTANTS
#undef X
// --------------------------------------------------------------------------------

class ErosionSimulator {

  private:
#ifdef ENABLE_EROSION_TRIPWIRE
    static bool instance_created; // tripwire to stop more than one class instance (defensive hack)
#endif

    float *dev_heightmap = nullptr;
    float *dev_water = nullptr;
    // float *dev_outflow = nullptr;
    float *dev_sediment = nullptr;

  public:
    float rain_rate = 0.01f;
    float evaporation_rate = 0.005f;
    float erosion_rate = 0.01f;
    float deposition_rate = 0.25f;
    int steps = 128;

    float slope_threshold = 0.1;

    // --------------------------------------------------------------------------------
    // Declare CUDA constant get/sets
#define X(TYPE, NAME, DEFAULT_VAL)        \
  public:                                 \
    TYPE get_##NAME() const;              \
    void set_##NAME(const TYPE p_##NAME); \
                                          \
  private:                                \
    TYPE NAME##_host = DEFAULT_VAL;
    EROSION_CONSTANTS
#undef X
    // --------------------------------------------------------------------------------

  public:
    void run_erosion(float *host_data, int width, int height);

    ErosionSimulator();
    ~ErosionSimulator();
};

} // namespace erosion