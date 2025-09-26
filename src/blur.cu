/*

gaussian blur 1D kernel

*/
#include "core_api.h"
#include <cuda_runtime.h>


void buildGaussianKernel1D(float *kernel, int kSize, float sigma)
{
    float sum = 0.0f;
    int radius = kSize / 2;
    for (int i = -radius; i <= radius; ++i)
    {
        float value = expf(-(i * i) / (2 * sigma * sigma));
        kernel[i + radius] = value;
        sum += value;
    }
    for (int i = 0; i < kSize; ++i)
    {
        kernel[i] /= sum;
    }
}


__global__ void gaussianBlurHorizontal(const float *input, float *output, int width, int height, const float *kernel, int kSize, bool wrap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int radius = kSize / 2;
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k)
    {
        int ix = x + k;
        if (wrap)
            ix = (ix + width) % width;
        else
            ix = min(max(ix, 0), width - 1);

        float pixel = input[y * width + ix];
        float weight = kernel[k + radius];
        sum += pixel * weight;
    }
    output[y * width + x] = sum;
}

__global__ void gaussianBlurVertical(const float *input, float *output, int width, int height, const float *kernel, int kSize, bool wrap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int radius = kSize / 2;
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k)
    {
        int iy = y + k;
        if (wrap)
            iy = (iy + height) % height;
        else
            iy = min(max(iy, 0), height - 1);

        float pixel = input[iy * width + x];
        float weight = kernel[k + radius];
        sum += pixel * weight;
    }
    output[y * width + x] = sum;
}


void run_blur(float *host_data, int width, int height, float sigma, bool wrap)
{
    int imageSize = width * height * sizeof(float);
    int kSize = static_cast<int>(std::ceil(6 * sigma)) | 1;

    float *h_kernel = new float[kSize];
    buildGaussianKernel1D(h_kernel, kSize, sigma);

    float *d_input, *d_temp, *d_output, *d_kernel;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_temp, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kSize * sizeof(float));

    cudaMemcpy(d_input, host_data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    gaussianBlurHorizontal<<<grid, block>>>(d_input, d_temp, width, height, d_kernel, kSize, wrap);
    gaussianBlurVertical<<<grid, block>>>(d_temp, d_output, width, height, d_kernel, kSize, wrap);

    cudaMemcpy(host_data, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
    cudaFree(d_kernel);
    delete[] h_kernel;
}

