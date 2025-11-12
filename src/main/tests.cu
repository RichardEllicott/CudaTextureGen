#include "tests.cuh"
#include <cstdio>
#include <cuda_runtime.h>

namespace tests {

void print_debug_info() {
    // Force CUDA runtime initialization
    cudaError_t init_err = cudaFree(0);
    if (init_err != cudaSuccess) {
        fprintf(stderr, "CUDA initialization failed: %s\n", cudaGetErrorString(init_err));
        fprintf(stderr, "This usually means no compatible driver/runtime is installed.\n");
        return;
    }

    // Driver version
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        printf("CUDA Driver Version: %d.%d\n",
               driver_version / 1000, (driver_version % 1000) / 10);
    } else {
        fprintf(stderr, "Failed to query CUDA driver version.\n");
    }

    // Runtime version
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        printf("CUDA Runtime Version: %d.%d\n",
               runtime_version / 1000, (runtime_version % 1000) / 10);
    } else {
        fprintf(stderr, "Failed to query CUDA runtime version.\n");
    }

    // Device count
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        fprintf(stderr, "No CUDA devices available or driver mismatch.\n");
        return;
    }

    printf("CUDA Devices Available: %d\n", device_count);

    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable devices detected. Check driver installation.\n");
    }
}

// tested pos mod that wraps the map
__device__ __forceinline__ int posmod(int value, int mod) {
    int result = value % mod;
    return result < 0 ? result + mod : result;
}

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);

    // // mod test
    // int mod = 4;
    // for (int i = 0; i < 16; i++) {

    //     int i2 = i - 8;
    //     // int i3 = i2 % mod;
    //     int i3 = posmod(i2, 4);

    //     printf("mod(%d, %d)=>%d\n", i2, mod, i3);
    // }
}

void cuda_hello() {

    // printf("🧙 testing printf...\n");
    // println("🧙 testing my print...");

    print_debug_info();

    printf("Hello from CPU\n");

    // Launch kernel with 5 threads
    // hello_kernel<<<1, 5>>>();
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // debugging
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}

/*

#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel(int* out) {
    int tid = threadIdx.x;
    out[tid] = tid * 10;
}

extern "C" void cuda_hello() {
    printf("Hello from CPU\n");

    int* d_out = nullptr;
    int* h_out = new int[5];

    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc(&d_out, 5 * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        delete[] h_out;
        return;
    }

    // Launch kernel
    hello_kernel<<<1, 5>>>(d_out);

    // Synchronize to ensure kernel has finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        delete[] h_out;
        return;
    }

    // Copy results back to host
    err = cudaMemcpy(h_out, d_out, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        delete[] h_out;
        return;
    }

    // Print results
    for (int i = 0; i < 5; ++i)
        printf("out[%d] = %d\n", i, h_out[i]);

    // Clean up
    cudaFree(d_out);
    delete[] h_out;
}




*/

} // namespace tests