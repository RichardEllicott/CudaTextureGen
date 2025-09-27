// // src/erosion.cu
// #include "erosion.h"
// #include <cuda_runtime.h>
// #include <algorithm>

#include "erosion.h"
#include <cuda_runtime.h>

__global__ void erosion_kernel(float* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        data[idx] = fmaxf(0.0f, data[idx] - 1.0f);
    }
}

void run_erosion(float* host_data, int width, int height, int steps) {
    size_t size = width * height * sizeof(float);
    float* dev_data = nullptr;

    // Allocate once
    cudaMalloc(&dev_data, size);
    cudaMemcpy(dev_data, host_data, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Loop on the host, but keep data on device
    for (int s = 0; s < steps; ++s) {
        erosion_kernel<<<grid, block>>>(dev_data, width, height);
    }
    cudaDeviceSynchronize();

    // Copy back once
    cudaMemcpy(host_data, dev_data, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_data);
}
