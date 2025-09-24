// #include <cuda_runtime.h>

// __global__ void scale_kernel(float* data, int n, float factor) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         data[idx] *= factor;
//     }
// }

// extern "C" void scale_array(float* data, int n, float factor) {
//     float* d_data;
//     cudaMalloc(&d_data, n * sizeof(float));
//     cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

//     int threads = 256;
//     int blocks = (n + threads - 1) / threads;
//     scale_kernel<<<blocks, threads>>>(d_data, n, factor);

//     cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaFree(d_data);
// }



#include <cuda_runtime.h>  // Includes CUDA runtime API functions

// GPU kernel: scales each element of the array by a given factor
__global__ void scale_kernel(float* data, int n, float factor) {
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check to avoid accessing out-of-range memory
    if (idx < n) {
        data[idx] *= factor;  // Scale the element in-place
    }
}

// Host function callable from C or Python (via ctypes, nanobind, etc.)
extern "C" void scale_array(float* data, int n, float factor) {
    float* d_data;  // Pointer for device memory

    // Allocate memory on the GPU for n floats
    cudaMalloc(&d_data, n * sizeof(float));

    // Copy input data from host (CPU) to device (GPU)
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int threads = 256;  // Number of threads per block
    int blocks = (n + threads - 1) / threads;  // Number of blocks (ceil division)

    // Launch the kernel on the GPU
    scale_kernel<<<blocks, threads>>>(d_data, n, factor);

    // Copy the result back from device to host
    cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_data);
}





