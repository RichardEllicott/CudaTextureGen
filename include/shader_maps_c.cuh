/*



*/
#pragma once

// #include "core.h"
// #include <chrono>
#include <cuda_runtime.h>
#include <iostream>

namespace shader_maps_c {

class ShaderMaps {
  private:
  public:
    void generate_normal_map(
        const float *host_in, float *host_out,
        int width, int height,
        float normal_scale, bool wrap);
};
} // namespace shader_maps_c