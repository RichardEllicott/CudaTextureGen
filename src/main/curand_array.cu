#include "cuda/curand_array.cuh"

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

void CurandArray::init(size_t width, size_t height, dim3 grid, dim3 block, cudaStream_t stream) {

    rng_states.resize(width * height);

    // resize and allocate
    init_rand_states<<<grid, block, 0, stream>>>(rng_states.dev_ptr(), 1234UL, width, height); // init the rand states with a seed
    cudaStreamSynchronize(stream);

    // init_rand_states<<<grid, block>>>(rng_states.dev_ptr(), 1234UL, width, height);
    // cudaDeviceSynchronize();
}

void CurandArray::init(size_t width, size_t height, cudaStream_t stream) {

    // calculate grid and block
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rng_states.resize(width * height);
    init_rand_states<<<grid, block, 0, stream>>>(rng_states.dev_ptr(), 1234UL, width, height); // init the rand states with a seed
    cudaStreamSynchronize(stream);
}

} // namespace core::cuda