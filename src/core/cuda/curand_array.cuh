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
    void init(size_t width, size_t height, cudaStream_t stream);

    CurandArray() {
    }

    CurandArray(size_t width, size_t height, cudaStream_t stream) {
        init(width, height, stream);
    }

    curandState *dev_ptr() {
        return rng_states.dev_ptr();
    }
};

} // namespace core::cuda