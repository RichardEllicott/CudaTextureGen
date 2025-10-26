#include "erosion.cuh"
#include <curand_kernel.h>

namespace erosion {

#pragma region OLD_FOLD

// positive modulo wrap (note it might be faster to concider other wrap methods)
__device__ __forceinline__ int posmod(int value, int mod) {
    int result = value % mod;
    return result < 0 ? result + mod : result;
}

// initialize random gen (for all tiles)
__global__ void init_rand(curandState *states, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    curand_init(1234, idx, 0, &states[idx]);
}

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

        if (pars->jitter && pars->jitter > 0.0f) {
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

// Addressing helper
__device__ __forceinline__ int image_position_to_index(int x, int y, const int width, const int height, const bool wrap) {
    if (wrap) {
        x = (x % width + width) % width;
        y = (y % height + height) % height;
    } else {
        x = min(max(x, 0), width - 1);
        y = min(max(y, 0), height - 1);
    }
    return y * width + x;
}

// 🚧 not really too clever
__global__ void erode_kernel_02(
    Parameters *pars,
    int width, int height,
    float *heightmap, float *sediment, float *water_map,
    curandState *rand_states

) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    float h = heightmap[idx];
    float s = sediment[idx];
    float w = water_map[idx];

    w += pars->rain_rate;

    // 8-way neighbor offsets
    const int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    float total_slope = 0.0f;
    float slopes[8] = {0};

    // Compute slopes to neighbors
    for (int i = 0; i < 8; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];

        if (pars->wrap) {
            nx = (nx + width) % width;
            ny = (ny + height) % height;
        } else {
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;
        }

        int nIdx = ny * width + nx;
        float nh = heightmap[nIdx] + water_map[nIdx]; // total height includes water
        float slope = (h + w) - nh;

        if (pars->jitter && pars->jitter > 0.0f) {
            float rand = curand_uniform(&rand_states[idx]);
            slope += rand * pars->jitter;
        }

        if (slope > pars->slope_threshold) {
            slopes[i] = slope;
            total_slope += slope;
        }
    }

    // Erode based on water and slope
    float eroded = w * pars->erosion_rate * total_slope;
    h -= eroded;
    s += eroded;

    // Distribute sediment and water to neighbors
    if (total_slope > 0.0f) {
        for (int i = 0; i < 8; ++i) {
            if (slopes[i] > 0) {
                int nx = x + dx[i];
                int ny = y + dy[i];

                if (!pars->wrap && (nx < 0 || nx >= width || ny < 0 || ny >= height))
                    continue;

                int nIdx = ((ny + height) % height) * width + ((nx + width) % width);

                float slope_ratio = slopes[i] / total_slope;

                float sediment_share = slope_ratio * pars->deposition_rate * s;
                float water_share = slope_ratio * w * pars->flow_factor;

                atomicAdd(&heightmap[nIdx], sediment_share);
                atomicAdd(&sediment[nIdx], -sediment_share);
                atomicAdd(&water_map[nIdx], water_share);
            }
        }
    }

    // Evaporation or absorption
    w *= (1.0f - pars->evaporation_rate);

    // Write back
    heightmap[idx] = h;
    sediment[idx] = s;
    water_map[idx] = w;
}

#pragma endregion

//
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

// downhill slope calculation
#if 0
    int idx = y * width + x;
    float h = height_map[idx];

    // Offsets, axis first, then diagonals
    const int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    float dir_x = 0.0f; // flow direction x
    float dir_y = 0.0f;

    for (int i = 0; i < 8; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height)
            continue;

        int nIdx = ny * width + nx;
        float nh = height_map[nIdx];
        float slope = h - nh;

        if (slope > 0.0f) {
            dir_x += dx[i] * slope;
            dir_y += dy[i] * slope;
        }
    }
#endif

// sobel pattern
#if 0 
    // Offsets clockwise from top
    const int ox[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    const int oy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

    float samples[8];
    for (int i = 0; i < 8; ++i) {
        int nx = x + ox[i];
        int ny = y + oy[i];
        int idx = image_position_to_index(nx, ny, width, height, true);
        samples[i] = height_map[idx];
    }

    // Sobel operator
    float dx = (samples[1] + 2 * samples[2] + samples[3]) - (samples[7] + 2 * samples[6] + samples[5]);
    float dy = (samples[5] + 2 * samples[4] + samples[3]) - (samples[7] + 2 * samples[0] + samples[1]);
#endif

    int idx = y * width + x;
    float h = height_map[idx];
    float w = water_map[idx];

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
    if (total_slope > 0.0f && w > 0.0f) {
        for (int i = 0; i < 8; ++i) {
            float share = (slope[i] / total_slope) * w; // share of water
            // outflow.x += dx[i] * share;
            // outflow.y += dy[i] * share;
        }
    }

    //   // Step 3: Store outflow vector
    // outflow_x[idx] = outflow.x;
    // outflow_y[idx] = outflow.y;
}

void Erosion::run_erosion(float *host_data) {

    printf("Erosion::run_erosion()...\n");

    size_t size = width * height * sizeof(float);
    dim3 block(pars.block_size, pars.block_size);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    curandState *dev_rand_states = nullptr;
    if (pars.jitter > 0.0f) {
        CUDA_CHECK(cudaMalloc(&dev_rand_states, width * height * sizeof(curandState)));
        init_rand<<<grid, block>>>(dev_rand_states, width, height);
    }

    allocate_device_memory();
    CUDA_CHECK(cudaMemcpy(dev_height_map, host_data, size, cudaMemcpyHostToDevice)); // copy data
    CUDA_CHECK(cudaMemset(dev_sediment_map, 0, size));                               // 0
    CUDA_CHECK(cudaMemset(dev_water_map, 0, size));                                  // 0

    // ⏲️ Timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Loop on the host, but keep data on device
    for (int s = 0; s < pars.steps; ++s) {

        switch (pars.mode) {
        case 0: // basic erode (no water)
            erode_kernel_01<<<grid, block>>>(
                dev_pars, width, height,
                dev_height_map, dev_sediment_map,
                dev_rand_states);
            break;

        case 1:
            erode_kernel_02<<<grid, block>>>(
                dev_pars, width, height,
                dev_height_map, dev_sediment_map, dev_water_map,
                dev_rand_states);
            break;
        case 2:

            my_erode_kernel_01<<<grid, block>>>(
                dev_pars, width, height,
                dev_height_map, dev_sediment_map, dev_water_map);

            break;
        }
    }
    cudaDeviceSynchronize();

    // ⏲️ Timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double seconds = elapsed.count();
    core::println("calculation time: ", seconds * 1000.0, " ms\n"); // ⏱️

    // Copy back once
    cudaMemcpy(host_data, dev_height_map, size, cudaMemcpyDeviceToHost);

    copy_maps_from_device(); // 🚧 new method to get data back

    free_device_memory();

    if (dev_rand_states) {
        CUDA_CHECK(cudaFree(dev_rand_states));
    }
}

} // namespace erosion