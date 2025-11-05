/*

random wrapper

*/
#pragma once
#include "cuda/device_array.cuh"
#include <curand_kernel.h>

// #include "types.h"

namespace core::cuda {



// __global__ void generate_noise(float *output, curandState *states) {
//     // int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     // curandState local_state = states[idx];

//     // float r = curand_uniform(&local_state); // [0,1)
//     // output[idx] = r;

//     // states[idx] = local_state; // Save updated state
// }

class CurandArray {

  private:
    core::cuda::DeviceArray<curandState> rng_states;

  public:
    void init(size_t width, size_t height, dim3 block, dim3 grid, cudaStream_t stream);
    
    
    // {
    //     // rng_states.resize(size);                                                    // resize and allocate
    //     // init_rand_states<<<block, grid, 0, stream>>>(rng_states.dev_ptr(), 1234UL); // init the rand states with a seed
    // }

    curandState *dev_ptr() {
        return rng_states.dev_ptr();
    }

    //   public:
    //     CurandArray(size_t n, unsigned long seed)
    //         : size_(n) {

    //         // cudaMalloc(&states_, n * sizeof(curandState));
    //         // dim3 block(256);
    //         // dim3 grid((n + block.x - 1) / block.x);
    //         // init_rand_states<<<grid, block>>>(states_, seed);
    //         // cudaDeviceSynchronize();
    //     }

    //     ~CurandArray() {
    //         if (states_)
    //             cudaFree(states_);
    //     }

    //     // // Generate uniform noise into a device buffer
    //     // void generate(float *d_output) {
    //     //     dim3 block(256);
    //     //     dim3 grid((size_ + block.x - 1) / block.x);
    //     //     generate_noise<<<grid, block>>>(d_output, states_);
    //     // }

    //     size_t size() const { return size_; }

    //     curandState * dev_ptr(){
    //         return states_;
    //     }

    //   private:
    //     curandState *states_{nullptr};
    //     size_t size_{0};
};

} // namespace core::cuda