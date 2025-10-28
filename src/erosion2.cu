#include "erosion2.cuh"

#include <curand_kernel.h>

#include "simple_erode.cuh"

// #include "core.h"
#include <chrono>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>

namespace TEMPLATE_NAMESPACE {

#pragma region WORKING_ORGINAL

// working basic erode, has no water just sediment redistribution
__global__ void erode_kernel_01(
    Parameters *pars,
    int width, int height,
    float *heightmap, float *sediment,
    curandState *rand_states

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

        if (pars->wrap) {
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

        if (rand_states && pars->jitter && pars->jitter > 0.0f) {
            float rand = curand_uniform(&rand_states[idx]); // [0,1)
            slope += rand * pars->jitter;
        }

        if (slope > pars->slope_threshold) {
            slopes[i] = slope;
            total_slope += slope;
        }
    }

    // Erode and deposit based on slope
    float eroded = pars->erosion_rate * total_slope;
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
            float share = (slopes[i] / total_slope) * pars->deposition_rate * s;

            // Atomic to avoid race conditions
            atomicAdd(&heightmap[nIdx], share);
            atomicAdd(&sediment[nIdx], -share);
        }
    }

    // Write back
    heightmap[idx] = h;
    sediment[idx] = s;
}

#pragma endregion

// positive modulo wrap (note it might be faster to concider other wrap methods)
__device__ __forceinline__ int posmod(int value, int mod) {
    int result = value % mod;
    return result < 0 ? result + mod : result;
}

// 🚧 my personal design
__global__ void my_erode_kernel_01(
    Parameters *pars,
    int width, int height,
    float *height_map, float *sediment_map, float *water_map) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // OPTIONAL STRUCTURES
    // const int2 offsets[8] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    // float2 dir = make_float2(0.0f, 0.0f);
    // const float2 dir = {0.0f, 0.0f};

    int idx = y * width + x;
    float h = height_map[idx];
    float water = water_map[idx];

    water += pars->rain_rate;

    // const int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    // const int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    const int2 offsets[8] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

    float total_slope = 0.0f;
    float slope[8];

    // Step 1: Compute slopes to downhill neighbors
    for (int i = 0; i < 8; ++i) {
        // int nx = x + dx[i]; // offset x
        // int ny = y + dy[i]; // offset y

        int nx = x + offsets[i].x; // offset x
        int ny = y + offsets[i].y; // offset y

        if (pars->wrap) {
            nx = posmod(nx, width);  // wrapped mod
            ny = posmod(ny, height); // wrapped mod

        } else {
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                slope[i] = 0.0f;
                continue;
            }
        }

        int n_idx = ny * width + nx; // new idx
        float nh = height_map[n_idx];
        float s = h - nh;
        slope[i] = (s > 0.0f) ? s : 0.0f;
        total_slope += slope[i];
    }

    // Step 2: Distribute water proportionally
    // float2 outflow = make_float2(0.0f, 0.0f);
    if (total_slope > 0.0f && water > 0.0f) {

        for (int i = 0; i < 8; ++i) {
            float share = (slope[i] / total_slope) * water; // share of water
            // outflow.x += dx[i] * share;
            // outflow.y += dy[i] * share;

            // 🚧 NEW BIT 🚧
            int nx = x + offsets[i].x; // offset x
            int ny = y + offsets[i].y; // offset y
            if (pars->wrap) {
                nx = posmod(nx, width);  // wrapped mod
                ny = posmod(ny, height); // wrapped mod
            } else {
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    slope[i] = 0.0f;
                    continue;
                }
            }
            int n_idx = ny * width + nx; // new idx
            atomicAdd(&water_map[n_idx], share);
            // atomicAdd(&sediment[nIdx], -share);
        }
    }

    water_map[idx] = water;

    // // Step 3: Store outflow vector
    // outflow_x[idx] = outflow.x;
    // outflow_y[idx] = outflow.y;
}

void TEMPLATE_CLASS_NAME::process() {

    core::CudaStream stream; // create a stream

    height_map.upload();
    pars.width = height_map.get_width();
    pars.height = height_map.get_height();

    sediment_map.resize(pars.width, pars.height);
    sediment_map.clear(); // ensure zeros (i think required)
    sediment_map.upload();

    water_map.resize(pars.width, pars.height);
    water_map.clear(); // ensure zeros (i think required)
    water_map.upload();

    // #define X(TYPE, NAME) \
    //     NAME.upload_to_device();
    //     TEMPLATE_CLASS_MAPS
    // #undef X

    core::CudaStruct<Parameters> gpu_pars(pars); // automaticly uploads and free

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    auto start_time = std::chrono::high_resolution_clock::now(); // ⏲️ Timer

    for (int s = 0; s < pars.steps; ++s) {
        switch (pars.mode) {
        case 0:
            erode_kernel_01<<<grid, block, 0, stream.get()>>>(gpu_pars.device_ptr(), pars.width, pars.height,
                                                              height_map.device_ptr(),
                                                              sediment_map.device_ptr(),
                                                              nullptr);
            break;
        case 1:
            my_erode_kernel_01<<<grid, block, 0, stream.get()>>>(gpu_pars.device_ptr(), pars.width, pars.height,
                                                                 height_map.device_ptr(),
                                                                 sediment_map.device_ptr(),
                                                                 water_map.device_ptr());

            break;

        case 2:
            simple_erode<<<grid, block, 0, stream.get()>>>(pars.width, pars.height,
                                                           height_map.device_ptr(),
                                                           sediment_map.device_ptr(),
                                                           nullptr,
                                                           pars.wrap,
                                                           pars.jitter,
                                                           pars.erosion_rate,
                                                           pars.slope_threshold,
                                                           pars.deposition_rate);

            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now(); // ⏲️ Timer
    std::chrono::duration<double> elapsed = end_time - start_time;
    double seconds = elapsed.count();
    printf("calculation time: %.2f ms\n", seconds * 1000.0); // ⏱️

    height_map.download();
    height_map.free_device();

    sediment_map.download();
    sediment_map.free_device();

    water_map.download();
    water_map.free_device();

    stream.sync(); // sync the stream

    // #define X(TYPE, NAME)            \
    //     NAME.download_from_device(); \
    //     NAME.free_device_memory();   \
    //     TEMPLATE_CLASS_MAPS
    // #undef X
}

} // namespace TEMPLATE_NAMESPACE
