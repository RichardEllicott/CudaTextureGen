#include "erosion2.cuh"
#include <assert.h> // optional


#define ENABLE_EROSION_WRAP

namespace erosion2 {

// __global__ void erode_kernel(const Parameters *pars, Maps *maps, const int width, const int height) {
// }

__global__ void erode_kernel(
    const Parameters *pars, Maps *maps,

    int width, int height


) {



    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    // 🚧🚧🚧
    float* heightmap = maps->height_map;
    float* sediment = maps->sediment_map;
    auto slope_threshold = pars->slope_threshold;
    auto erosion_rate = pars->erosion_rate;
    auto deposition_rate = pars->deposition_rate;
    // 🚧🚧🚧



    float h = heightmap[idx];
    float s = sediment[idx];

    // 8-way neighbor offsets
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    float total_slope = 0.0f;
    float slopes[8] = {0};

    // Compute slopes to neighbors
    for (int i = 0; i < 8; ++i) {

#ifdef ENABLE_EROSION_WRAP
        int nx = (x + dx[i] + width) % width;
        int ny = (y + dy[i] + height) % height;
#else
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height)
            continue;
#endif

        int nIdx = ny * width + nx;
        float nh = heightmap[nIdx];
        float slope = h - nh;



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

void Erosion2::process() {

    printf("HSHSHSHHSHHSHS......!!!.....");

    // ⚠️ we need to have set our data ptr in

    allocate_and_copy_to_gpu();

    size_t map_size = host_pars.width * host_pars.height; // find map size

    // 🚧 we could use async here
    CUDA_CHECK(cudaMalloc(&device_map_pointers.water_map, map_size * sizeof(float)));
    cudaMemset(device_map_pointers.water_map, 0, map_size); // start with no water
    CUDA_CHECK(cudaMalloc(&device_map_pointers.sediment_map, map_size * sizeof(float)));
    cudaMemset(device_map_pointers.sediment_map, 0, map_size); // start with no sediment

    dim3 block(16, 16);
    dim3 grid((host_pars.width + block.x - 1) / block.x,
              (host_pars.height + block.y - 1) / block.y);

    // Launch kernel with access to private members
    erode_kernel<<<grid, block>>>(device_pars, device_map_struct, host_pars.width, host_pars.height);

    CUDA_CHECK(cudaGetLastError());      // Check launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Check execution

    if (device_map_pointers.water_map) {
        cudaFree(device_map_pointers.water_map);
        device_map_pointers.water_map = nullptr;
    }

    if (device_map_pointers.sediment_map) {
        cudaFree(device_map_pointers.sediment_map);
        device_map_pointers.sediment_map = nullptr;
    }

    copy_maps_back_from_gpu();

    free_memory();
}

} // namespace erosion2