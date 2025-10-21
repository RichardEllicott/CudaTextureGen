/*



*/
#pragma once

// #include "core.h"
// #include <chrono>
#include <cuda_runtime.h>
#include <iostream>

namespace shader_maps {

class ShaderMaps {
  private:
  public:
    void generate_normal_map(
        const float *host_in, float *host_out,
        int width, int height,
        float scale, bool wrap);

    void generate_ao_map(
        const float *host_in, float *host_out,
        int width, int height,
        int radius, bool wrap, int mode = 0);
};
} // namespace shader_maps