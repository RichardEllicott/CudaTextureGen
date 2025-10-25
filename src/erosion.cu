#include "erosion.cuh"

namespace erosion {

#ifdef ENABLE_EROSION_TRIPWIRE
bool Erosion::instance_created = false; // set tripwire
#endif

#pragma region MAIN

#ifdef ENABLE_EROSION_JITTER
__global__ void initRand(curandState *states, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    curand_init(1234, idx, 0, &states[idx]);
}
#endif

__global__ void erode_kernel(
    Parameters *pars,
    int width, int height,

    float *heightmap, float *sediment

#ifdef ENABLE_EROSION_JITTER
    ,
    curandState *rand_states
#endif

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

#ifdef ENABLE_EROSION_JITTER

        if (pars->jitter > 0.0f) {
            float rand = curand_uniform(&rand_states[idx]); // [0,1)
            slope += rand * pars->jitter;
        }

#endif

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

#pragma region CLASS

// NOT REQUIRED
//         cudaMemcpyToSymbol(erosion::NAME, &p_##NAME, sizeof(TYPE)); \


void Erosion::run_erosion(float *host_data, int width, int height) {

    size_t size = width * height * sizeof(float);

#ifdef ENABLE_EROSION_JITTER
    curandState *dev_rand_states;
    CUDA_CHECK(
        cudaMalloc(&dev_rand_states, width * height * sizeof(curandState)));
#endif

    // copy pars to gpu
    CUDA_CHECK(cudaMalloc(&dev_pars, sizeof(Parameters)));
    CUDA_CHECK(cudaMemcpy(dev_pars, &pars, sizeof(Parameters), cudaMemcpyHostToDevice));

    // allocate memory
    CUDA_CHECK(cudaMalloc(&dev_heightmap, size));
    CUDA_CHECK(cudaMemcpy(dev_heightmap, host_data, size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dev_water, size));
    CUDA_CHECK(cudaMemset(dev_water, 0, size)); // start with no water

    // cudaMalloc(&dev_outflow, size);
    // cudaMemset(dev_outflow, 0, size);

    cudaMalloc(&dev_sediment, size);
    cudaMemset(dev_sediment, 0, size);

    dim3 block(pars.block_size, pars.block_size);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // ⏲️ Timer
    auto start_time = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_EROSION_JITTER
    initRand<<<grid, block>>>(dev_rand_states, width, height);
#endif

    // Loop on the host, but keep data on device
    for (int s = 0; s < pars.steps; ++s) {

        // rain_kernel<<<grid, block>>>(dev_water, width, height, rain_rate);
        // flow_kernel<<<grid, block>>>(dev_height, dev_water, dev_outflow, width, height);
        // erosion_kernel<<<grid, block>>>(dev_height, dev_water, dev_sediment, width, height, erosion_rate, deposition_rate);

        // sediment_transport_kernel<<<grid, block>>>(dev_height, dev_water, dev_sediment, width, height); // new

        // evaporation_kernel<<<grid, block>>>(dev_water, width, height, evaporation_rate);

        erode_kernel<<<grid, block>>>(
            dev_pars, width, height,
            dev_heightmap, dev_sediment

#ifdef ENABLE_EROSION_JITTER
            ,
            dev_rand_states
#endif
        );
    }
    cudaDeviceSynchronize();

    // ⏲️ Timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double seconds = elapsed.count();
    println("calculation time ⏱️: ", seconds * 1000.0, " ms");

    // Copy back once
    cudaMemcpy(host_data, dev_heightmap, size, cudaMemcpyDeviceToHost);

    // // free data
    CUDA_CHECK(cudaFree(dev_pars));

    CUDA_CHECK(cudaFree(dev_heightmap));
    CUDA_CHECK(cudaFree(dev_water));
    CUDA_CHECK(cudaFree(dev_sediment));
}

Erosion::Erosion() {

#ifdef ENABLE_EROSION_TRIPWIRE
    if (instance_created) {
        std::cerr << "ERROR: TerrainEroder already instantiated!" << std::endl;
        std::abort(); // or throw std::runtime_error
    }
    instance_created = true;
#endif
}

Erosion::~Erosion() {
#ifdef ENABLE_EROSION_TRIPWIRE
    instance_created = false;
#endif
}

#pragma endregion

} // namespace erosion