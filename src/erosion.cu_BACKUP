#include "erosion.cuh"

#define EROSION_BLOCK_SIZE 16 // normally the best size for block, 16x16 = 256 threads per block 8 warps (32 threads a warp)
// #define EROSION_BLOCK_SIZE 8 // smaller blocks (not a good idea)

namespace erosion {

#ifdef ENABLE_EROSION_TRIPWIRE
bool ErosionSimulator::instance_created = false; // set tripwire
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
    float *heightmap, float *sediment,
    int width, int height,
    float erosion_rate, float deposition_rate, float slope_threshold
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

#ifdef ENABLE_EROSION_JITTER
        // float jitter = 0.01f * (rand() % 1000) / 1000.0f;
        // slope += jitter;
        float rand = curand_uniform(&rand_states[idx]); // [0,1)
        slope += rand * jitter;
#endif

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

#pragma endregion

#pragma region CLASS

// --------------------------------------------------------------------------------
// Define CUDA constant get/sets
#define X(TYPE, NAME, DEFAULT_VAL)                           \
    TYPE ErosionSimulator::get_##NAME() const {              \
        return NAME##_host;                                  \
    }                                                        \
    void ErosionSimulator::set_##NAME(const TYPE p_##NAME) { \
        NAME##_host = p_##NAME;                              \
    }
EROSION_CONSTANTS
#undef X
// --------------------------------------------------------------------------------

// NOT REQUIRED
//         cudaMemcpyToSymbol(erosion::NAME, &p_##NAME, sizeof(TYPE)); \


void ErosionSimulator::run_erosion(float *host_data, int width, int height) {

    println("⛰️ run_erosion...");
    // println();
    // println("rain_rate = ", rain_rate, "");
    // println("evaporation_rate = ", evaporation_rate, "");
    println("erosion_rate = ", erosion_rate, "");
    println("deposition_rate = ", deposition_rate, "");
    println("slope_threshold = ", slope_threshold, "");
    // println();

    // println("steps = ", steps, "");

    // --------------------------------------------------------------------------------
    // print values
#define X(TYPE, NAME, DEFAULT_VAL) \
    println(#NAME, " = ", get_##NAME());
    EROSION_CONSTANTS
#undef X
    // --------------------------------------------------------------------------------

    // --------------------------------------------------------------------------------
    // Sync CUDA constants to device
#define X(TYPE, NAME, DEFAULT_VAL) \
    cudaMemcpyToSymbol(erosion::NAME, &NAME##_host, sizeof(TYPE));
    EROSION_CONSTANTS
#undef X
    // --------------------------------------------------------------------------------

    println();

    size_t size = width * height * sizeof(float);

#ifdef ENABLE_EROSION_JITTER
    curandState *dev_rand_states;
    cudaMalloc(&dev_rand_states, width * height * sizeof(curandState));
#endif

    // allocate memory
    cudaMalloc(&dev_heightmap, size);
    cudaMemcpy(dev_heightmap, host_data, size, cudaMemcpyHostToDevice);

    cudaMalloc(&dev_water, size);
    cudaMemset(dev_water, 0, size); // start with no water

    // cudaMalloc(&dev_outflow, size);
    // cudaMemset(dev_outflow, 0, size);

    cudaMalloc(&dev_sediment, size);
    cudaMemset(dev_sediment, 0, size);

    dim3 block_size(EROSION_BLOCK_SIZE, EROSION_BLOCK_SIZE);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // ⏲️ Timer
    auto start_time = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_EROSION_JITTER
    initRand<<<grid_size, block_size>>>(dev_rand_states, width, height);
#endif

    // Loop on the host, but keep data on device
    for (int s = 0; s < steps; ++s) {

        // rain_kernel<<<grid, block>>>(dev_water, width, height, rain_rate);
        // flow_kernel<<<grid, block>>>(dev_height, dev_water, dev_outflow, width, height);
        // erosion_kernel<<<grid, block>>>(dev_height, dev_water, dev_sediment, width, height, erosion_rate, deposition_rate);

        // sediment_transport_kernel<<<grid, block>>>(dev_height, dev_water, dev_sediment, width, height); // new

        // evaporation_kernel<<<grid, block>>>(dev_water, width, height, evaporation_rate);

        erode_kernel<<<grid_size, block_size>>>(
            dev_heightmap, dev_sediment,
            width, height,
            erosion_rate, deposition_rate, slope_threshold
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
    cudaFree(dev_heightmap);
    cudaFree(dev_water);
    // cudaFree(dev_outflow);
    cudaFree(dev_sediment);
}

ErosionSimulator::ErosionSimulator() {

#ifdef ENABLE_EROSION_TRIPWIRE
    if (instance_created) {
        std::cerr << "ERROR: TerrainEroder already instantiated!" << std::endl;
        std::abort(); // or throw std::runtime_error
    }
    instance_created = true;
#endif
    // --------------------------------------------------------------------------------
    // Sync CUDA constants to device
#define X(TYPE, NAME, DEFAULT_VAL) \
    set_##NAME(DEFAULT_VAL);
    EROSION_CONSTANTS
#undef X
    // --------------------------------------------------------------------------------
}

ErosionSimulator::~ErosionSimulator() {
#ifdef ENABLE_EROSION_TRIPWIRE
    instance_created = false;
#endif
}

#pragma endregion

} // namespace erosion