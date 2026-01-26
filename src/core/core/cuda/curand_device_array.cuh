/*

random wrapper

*/
#pragma once

#include "device_array.cuh"
#include <curand_kernel.h>

namespace core::cuda {

template <int Dim>
class CurandDeviceArray {

  private:
    DeviceArray<curandState, Dim> device_array;

  public:
    unsigned long seed = 1234UL;
    // compute the seeds
    void init();

    void resize(std::array<size_t, Dim> dimensions) {

        if (device_array.shape() != dimensions) {
            device_array.resize(dimensions);
            init();
        }
    }
};

using CurandDeviceArray2D = CurandDeviceArray<2>;

// // __global__ void generate_noise(float *output, curandState *states) {
// //     // int idx = threadIdx.x + blockIdx.x * blockDim.x;
// //     // curandState local_state = states[idx];

// //     // float r = curand_uniform(&local_state); // [0,1)
// //     // output[idx] = r;

// //     // states[idx] = local_state; // Save updated state
// // }

// // template <int Dim>
// // class CurandArray{

// // };

// template <int Dim>
// class CurandArray {

//   private:
//     core::cuda::DeviceArray<curandState, Dim> device_array;

//   public:
//     unsigned long seed = 1234UL;
//     // compute the seeds
//     void init();

//     void resize(std::array<size_t, Dim> dimensions) {

//         if (device_array.shape() != dimensions) {
//             device_array.resize(dimensions);
//             init();
//         }
//     }
// };

// class CurandArray2D {

//     core::cuda::DeviceArray2D<curandState> device_array;

//   public:
//     CurandArray2D() {
//     }

//     unsigned long seed = 1234UL;

//     // compute the seeds
//     void init();

//     // resize the array, and computer the seeds if required
//     void resize(size_t width, size_t height);

//     // free device, the same as setting the size to 0,0
//     void free_device() {
//         resize(0, 0);
//     }

//     curandState *dev_ptr() {
//         return device_array.dev_ptr();
//     }

//     cudaStream_t get_stream() const { return device_array.get_stream(); }
//     void set_stream(cudaStream_t stream) { device_array.set_stream(stream); }
// };

} // namespace core::cuda