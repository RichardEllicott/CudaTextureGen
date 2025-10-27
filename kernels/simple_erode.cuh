/*

saving a simple working erosion kernel

*/
#pragma once

#include <curand_kernel.h>

// working basic erode, has no water just sediment redistribution
__global__ void simple_erode(
    const int width, const int height,
    float *heightmap, float *sediment,
    curandState *rand_states = nullptr,
    const bool wrap = true,
    const float jitter = 0.0f,
    const float erosion_rate = 0.01f,
    const float slope_threshold = 0.01f,
    const float deposition_rate = 0.01f

) {

#ifdef ENABLE_EROSION_TILED_MEMORY
    __shared__ float tile[EROSION_BLOCK_SIZE + 2][EROSION_BLOCK_SIZE + 2]; // +2 for 1-cell border
#endif

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    float h = heightmap[idx];
    float s = sediment[idx];

    // 8-way neighbor offsets
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    float total_slope = 0.0f;
    float slopes[8] = {0};

    // Compute slopes to neighbors
    for (int i = 0; i < 8; ++i) {

        int nx;
        int ny;

        if (wrap) {
            nx = (x + dx[i] + width) % width;
            ny = (y + dy[i] + height) % height;
        } else {
            nx = x + dx[i];
            ny = y + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;
        }

        int nIdx = ny * width + nx;
        float nh = heightmap[nIdx];
        float slope = h - nh;

        if (rand_states && jitter && jitter > 0.0f) {
            float rand = curand_uniform(&rand_states[idx]); // [0,1)
            slope += rand * jitter;
        }

        if (slope > slope_threshold) {
            slopes[i] = slope;
            total_slope += slope;
        }
    }

    // Erode and deposit based on slope
    float eroded = erosion_rate * total_slope;
    h -= eroded;
    s += eroded;

    // Distribute sediment to neighbors
    for (int i = 0; i < 8; ++i) {
        if (slopes[i] > 0) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;

            int nIdx = ny * width + nx;
            float share = (slopes[i] / total_slope) * deposition_rate * s;

            // Atomic to avoid race conditions
            atomicAdd(&heightmap[nIdx], share);
            atomicAdd(&sediment[nIdx], -share);
        }
    }

    // Write back
    heightmap[idx] = h;
    sediment[idx] = s;
}

