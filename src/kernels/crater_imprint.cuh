/*

*/
#pragma once

namespace kernels {

// 🚧🚧🚧🚧🚧 UNTESTED AI GENERATED 🚧🚧🚧🚧🚧
//
// Apply inverse-square crater imprint onto a heightmap.
__global__ void crater_imprint(float *height_map, int map_width, int map_height,
                               float position_x, float position_y,
                               float excavation_scale, float softening_radius, float mask_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height)
        return;

    float dx = (x + 0.5f) - position_x;
    float dy = (y + 0.5f) - position_y;
    float r2 = dx * dx + dy * dy;

    if (mask_radius > 0.0f && r2 > mask_radius * mask_radius)
        return;

    float denom = r2 + softening_radius * softening_radius; // softening
    float E = 1.0f / denom;                                 // inverse-square
    float dh = -excavation_scale * E;                       // excavation depth

    // Optional: taper center to avoid a pixel spike when r0 is small
    // dh *= (r2 / (r2 + r0*r0));

    int idx = y * map_width + x;
    height_map[idx] += dh;
}

} // namespace kernels