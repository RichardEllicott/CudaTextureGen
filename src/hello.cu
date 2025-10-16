#include "core_api.h"
#include <cstdio>



void print_debug_info() {


    // Force runtime initialization
    cudaFree(0);

    // Get driver version
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    printf("CUDA Driver Version: %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);

    // Get runtime version
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    printf("CUDA Runtime Version: %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);

    // Get device count
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("CUDA Devices Available: %d\n", device_count);
}





__global__ void hello_kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

extern "C" void cuda_hello() {


    print_debug_info();


    printf("Hello from CPU\n");

    // Launch kernel with 5 threads
    hello_kernel<<<1, 5>>>();
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



