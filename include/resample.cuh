/*



*/
#pragma once

// #include "core.h"
// #include <chrono>
#include <cuda_runtime.h>
#include <iostream>

namespace resample {

class Resample {
  private:
  public:
    void process_maps(
        const float *host_in, float *host_out,
        const int width, const int height,
        const float *map_x, const float *map_y);
};
} // namespace resample