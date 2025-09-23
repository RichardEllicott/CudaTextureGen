#include <cuda_runtime.h>

__global__ void scale_kernel(float* data, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}

extern "C" void scale_array(float* data, int n, float factor) {
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(d_data, n, factor);

    cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
