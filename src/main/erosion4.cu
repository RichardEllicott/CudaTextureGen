#include "erosion4.cuh"
#include <curand_kernel.h>

namespace TEMPLATE_NAMESPACE {

// random
__global__ void init_rand_states(curandState *states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void generate_noise(float *output, curandState *states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState local_state = states[idx];

    float r = curand_uniform(&local_state); // [0,1)
    output[idx] = r;

    states[idx] = local_state; // Save updated state
}

// working basic erode, has no water just sediment redistribution
// tested with ping pong but makes no difference!!
__global__ void simple_erode(
    Parameters *pars,
    const int width, const int height,
    const float *heightmap, const float *sediment,
    float *heightmap_out, float *sediment_out, // seems to make no difference using ping pong
    curandState *rand_states = nullptr

) {

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
        float slope = h - nh; // amount higher than this neighbour

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
            atomicAdd(&heightmap_out[nIdx], share);
            atomicAdd(&sediment_out[nIdx], -share);
        }
    }

    // Write back
    heightmap_out[idx] = h;
    sediment_out[idx] = s;
}

void TEMPLATE_CLASS_NAME::process() {

    // upload heightmap
    height_map.upload();

    // set width/height
    pars._width = height_map.get_width();
    pars._height = height_map.get_height();

    // match sediment map
    sediment_map.resize(pars._width, pars._height);
    sediment_map.clear();
    sediment_map.upload();

    // allocate ping pong arrays
    core::cuda::DeviceArray1D<float> height_map_out;
    core::cuda::DeviceArray1D<float> sediment_map_out;
    height_map_out.resize(height_map.size());
    sediment_map_out.resize(height_map.size());

    // copy data to ping pong arrays (on gpu)
    size_t num_bytes = height_map.size() * sizeof(float);
    cudaMemcpy(height_map_out.dev_ptr(), height_map.dev_ptr(), num_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(sediment_map_out.dev_ptr(), sediment_map.dev_ptr(), num_bytes, cudaMemcpyDeviceToDevice);

    core::cuda::Stream stream;                  // create stream
    core::cuda::DeviceStruct<Parameters> _pars(pars); // upload pars

    // calculate grid and block
    dim3 block(pars._block, pars._block);
    dim3 grid((pars._width + block.x - 1) / block.x,
              (pars._height + block.y - 1) / block.y);

    // pointers to swap
    auto h1 = height_map.dev_ptr();
    auto h2 = height_map_out.dev_ptr();
    auto s1 = sediment_map.dev_ptr();
    auto s2 = sediment_map_out.dev_ptr();

    // init random if we need jitter
    core::cuda::DeviceArray1D<curandState> rng_states;
    if (pars.jitter > 0.0f) {
        rng_states.resize(height_map.size());                                             // resize and allocate
        init_rand_states<<<grid, block, 0, stream.get()>>>(rng_states.dev_ptr(), 1234UL); // init the rand states with a seed
    }

    switch (pars.mode) {
    case 0:
        printf("simple_erode...");
        for (int i = 0; i < pars.steps; i++) {
            simple_erode<<<grid, block, 0, stream.get()>>>(
                _pars.dev_ptr(),
                pars._width, pars._height,
                height_map.dev_ptr(), sediment_map.dev_ptr(),
                height_map.dev_ptr(), sediment_map.dev_ptr(),
                rng_states.dev_ptr());
        }
        break;
    case 1: // ping pong mode
        printf("simple_erode (ping pong mode)...");
        for (int i = 0; i < pars.steps; i++) {
            simple_erode<<<grid, block, 0, stream.get()>>>(
                _pars.dev_ptr(),
                pars._width, pars._height,
                h1, s1,
                h2, s2,
                rng_states.dev_ptr());
            std::swap(h1, h2);
            std::swap(s1, s2);
        }
        break;
    }

    stream.sync();

    height_map.download();
    height_map.free_device();

    sediment_map.download();
    sediment_map.free_device();
}

} // namespace TEMPLATE_NAMESPACE
