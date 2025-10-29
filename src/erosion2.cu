#include "erosion2.cuh"

#include <curand_kernel.h>

#include "simple_erode.cuh"

// #include "core.h"
#include <chrono>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>

namespace TEMPLATE_NAMESPACE {

__device__ __forceinline__ int wrap_or_clamp(int i, int n, bool wrap) {
    if (wrap) {
        int m = i % n;
        return m < 0 ? m + n : m;
    }
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// calculates the changes that need to occur
__global__ void flux_pass(
    const Parameters pars,
    int width, int height,
    const float *__restrict__ height_in,
    const float *__restrict__ water_in,
    const float *__restrict__ sediment_in,
    // outputs
    float *__restrict__ flux0, // 8 fluxes per cell (neighbor order)
    float *__restrict__ dh_out,
    float *__restrict__ ds_out,
    float *__restrict__ dw_out) {
    const int2 offs[8] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    const float dist[8] = {1, 1, 1, 1, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    float h = height_in[idx];
    float w = water_in[idx] + pars.rain_rate;
    float s_cur = sediment_in[idx];

    float slopes[8];
    float sum_slope = 0.f;

    // neighbor heights and slopes
    for (int i = 0; i < 8; ++i) {
        int nx = wrap_or_clamp(x + offs[i].x, width, pars.wrap);
        int ny = wrap_or_clamp(y + offs[i].y, height, pars.wrap);
        int nidx = ny * width + nx;
        float nh = height_in[nidx];
        float s = (h - nh) / dist[i];
        float sd = s > 0.f ? s : 0.f;
        slopes[i] = sd;
        sum_slope += sd;
    }

    float outflow_cap = fminf(w, pars.w_max);
    float outflow_sum = 0.f;

    // proportional flux
    float *cell_flux = &flux0[idx * 8];
    if (sum_slope > 1e-6f && outflow_cap > 0.f) {
        for (int i = 0; i < 8; ++i) {
            float q = (slopes[i] / sum_slope) * outflow_cap;
            cell_flux[i] = q;
            outflow_sum += q;
        }
    } else {
        for (int i = 0; i < 8; ++i)
            cell_flux[i] = 0.f;
    }

    // velocity proxy and capacity
    float v = 0.f;
    for (int i = 0; i < 8; ++i)
        v += cell_flux[i] * slopes[i];
    float C = pars.k_capacity * v;

    float erode = 0.f, deposit = 0.f;
    if (C > s_cur) {
        erode = pars.k_erode * (C - s_cur);
    } else {
        deposit = pars.k_deposit * (s_cur - C);
    }

    // write deltas (applied in pass B)
    dh_out[idx] = deposit - erode;              // positive = deposition raises height
    ds_out[idx] = erode - deposit;              // sediment increases when eroding, decreases when depositing
    dw_out[idx] = -outflow_sum - pars.evap * w; // water loss: outflow + evaporation
}

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
            // case 0:
            //     erode_kernel_01<<<grid, block, 0, stream.get()>>>(gpu_pars.device_ptr(), pars.width, pars.height,
            //                                                       height_map.device_ptr(),
            //                                                       sediment_map.device_ptr(),
            //                                                       nullptr);
            //     break;

        case 0:
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

        case 1:
            my_erode_kernel_01<<<grid, block, 0, stream.get()>>>(gpu_pars.device_ptr(), pars.width, pars.height,
                                                                 height_map.device_ptr(),
                                                                 sediment_map.device_ptr(),
                                                                 water_map.device_ptr());

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
