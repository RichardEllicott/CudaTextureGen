/*



*/
#pragma once

#include <cuda_runtime.h>
#include <iostream>

namespace shader_maps {

void generate_normal_map(
    const float *host_in, float *host_out,
    int width, int height,
    float scale, bool wrap);

void generate_ao_map(
    const float *host_in, float *host_out,
    int width, int height,
    int radius, bool wrap, int mode);

} // namespace shader_maps