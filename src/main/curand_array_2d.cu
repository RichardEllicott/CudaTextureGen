#include "core/cuda/curand_array_2d.cuh"

namespace core::cuda {

// // // random
// __global__ void init_rand_states(curandState *states, unsigned long seed) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     curand_init(seed, idx, 0, &states[idx]);
// }

// 2D init kernel
__global__ void init_rand_states(curandState *states,
                                 unsigned long seed,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;
    // sequence = idx, offset = 0
    curand_init(seed, idx, 0, &states[idx]);
}

// void CurandArray2D::init(size_t width, size_t height, dim3 grid, dim3 block, cudaStream_t stream) {

//     rng_states.resize(width * height);

//     init_rand_states<<<grid, block, 0, stream>>>(rng_states.dev_ptr(), 1234UL, width, height); // init the rand states with a seed
//     cudaStreamSynchronize(stream);
// }

void CurandArray2D::init(size_t width, size_t height, cudaStream_t stream) {

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rng_states.resize(width * height);
    init_rand_states<<<grid, block, 0, stream>>>(rng_states.dev_ptr(), 1234UL, width, height); // init the rand states with a seed
    // cudaStreamSynchronize(stream);
}

void CurandArray2D::init(size_t width, size_t height) {

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rng_states.resize(width * height);

    init_rand_states<<<grid, block>>>(rng_states.dev_ptr(), 1234UL, width, height); // init the rand states with a seed
    // cudaDeviceSynchronize();
}

// REFACTOR

void CurandArray2D_2::resize(size_t width, size_t height, cudaStream_t stream) {

    if (rng_states.width() != width || rng_states.height() != height) {

        rng_states.resize(width, height);

        if (!rng_states.empty()) {
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x,
                      (height + block.y - 1) / block.y);

            if (stream) {
                init_rand_states<<<grid, block, 0, stream>>>(rng_states.dev_ptr(), 1234UL, width, height); // init the rand states with a seed (using stream)
            } else {
                init_rand_states<<<grid, block>>>(rng_states.dev_ptr(), 1234UL, width, height); // init the rand states with a seed
            }
        }
    }
}

} // namespace core::cuda