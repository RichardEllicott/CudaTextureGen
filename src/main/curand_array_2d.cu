#include "core/cuda/curand_array_2d.cuh"

namespace core::cuda {

// alternate MAYBE 1D?
__global__ void init_rand_states(curandState *states,
                                 unsigned long seed,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    // sequence = idx, offset = 0
    curand_init(seed, idx, 0, &states[idx]);
}


template <int Dim>
void CurandArray<Dim>::init() {

    int N = device_array.size();
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    init_rand_states<<<blocks, threadsPerBlock>>>(this->states, this->seed, N); // added this-> here for linux?
}

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

void CurandArray2D::init() {

    if (!device_array.empty()) { // if we resized to a size that's not empty, generate the states

        size_t width = device_array.width();
        size_t height = device_array.height();

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        init_rand_states<<<grid, block, 0, get_stream()>>>(device_array.dev_ptr(), seed, width, height); // init the rand states with a seed (using stream)
    }
}

void CurandArray2D::resize(size_t width, size_t height) {
    if (device_array.width() != width || device_array.height() != height) { // if size changes
        device_array.resize(width, height);
        init();
    }
}

} // namespace core::cuda