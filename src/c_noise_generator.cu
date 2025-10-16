/*

started with co-pilot... solved very quick with claude:


https://claude.ai/chat/626e97e1-b6cb-4690-9dc7-db32b31ccccf


*/

#include "c_noise_generator.cuh"

namespace c_noise_generator {

#pragma region VALUE_NOISE

__device__ float hash(int x, int y, int seed) {
    int n = x + y * 57 + seed * 131;
    n = (n << 13) ^ n;
    return 1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f;
}

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Creates an S-curve (sigmoid-like shape)
__device__ float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ inline int posmod(int value, int mod) {
    int result = value % mod;
    return result < 0 ? result + mod : result;
}

__device__ float tileable_noise(float x, float y, int period_x, int period_y, int seed) {
    int xi = (int)floorf(x);
    int yi = (int)floorf(y);

    float xf = x - floorf(x);
    float yf = y - floorf(y);

    float u = fade(xf);
    float v = fade(yf);

    // Wrap grid coordinates to the period
    int xi0 = posmod(xi, period_x);
    int yi0 = posmod(yi, period_y);
    int xi1 = posmod(xi + 1, period_x);
    int yi1 = posmod(yi + 1, period_y);

    float a = hash(xi0, yi0, seed);
    float b = hash(xi1, yi0, seed);
    float c = hash(xi0, yi1, seed);
    float d = hash(xi1, yi1, seed);

    float x1 = lerp(a, b, u);
    float x2 = lerp(c, d, u);

    return lerp(x1, x2, v);
}

#pragma endregion

#pragma region GRADIENT_NOISE

__device__ float2 gradient(int x, int y, int seed) {
    int n = x + y * 57 + seed * 131;
    n = (n << 13) ^ n;
    n = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;

    // Convert to angle and create unit vector
    float angle = (n / 1073741824.0f) * 3.14159265f;
    return make_float2(cosf(angle), sinf(angle));
}

__device__ float gradient_noise(float x, float y, int period_x, int period_y, int seed) {
    int xi = (int)floorf(x);
    int yi = (int)floorf(y);

    float xf = x - floorf(x);
    float yf = y - floorf(y);

    float u = fade(xf);
    float v = fade(yf);

    int xi0 = posmod(xi, period_x);
    int yi0 = posmod(yi, period_y);
    int xi1 = posmod(xi + 1, period_x);
    int yi1 = posmod(yi + 1, period_y);

    // Get gradients at corners
    float2 g00 = gradient(xi0, yi0, seed);
    float2 g10 = gradient(xi1, yi0, seed);
    float2 g01 = gradient(xi0, yi1, seed);
    float2 g11 = gradient(xi1, yi1, seed);

    // Calculate dot products with distance vectors
    float d00 = g00.x * xf + g00.y * yf;
    float d10 = g10.x * (xf - 1.0f) + g10.y * yf;
    float d01 = g01.x * xf + g01.y * (yf - 1.0f);
    float d11 = g11.x * (xf - 1.0f) + g11.y * (yf - 1.0f);

    // Interpolate
    float x1 = lerp(d00, d10, u);
    float x2 = lerp(d01, d11, u);

    return lerp(x1, x2, v) * 0.5f + 0.5f; // Remap to [0,1]
}

#pragma endregion

#pragma region WARPED_NOISE

__device__ float warped_noise(float x, float y, int period, int seed) {
    // Sample noise at offset positions
    float warp_x = tileable_noise(x + 5.2f, y + 1.3f, period, period, seed + 1) * 4.0f;
    float warp_y = tileable_noise(x + 3.7f, y + 9.1f, period, period, seed + 2) * 4.0f;

    // Use warped coordinates for final sample
    return tileable_noise(x + warp_x, y + warp_y, period, period, seed);
}

#pragma endregion

__global__ void generate_noise(float *out, int width, int height, float scale, int period, int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Key fix: scale determines the noise frequency
    // The period should match the scaled coordinate space
    float fx = x * scale;
    float fy = y * scale;

    int idx = y * width + x;

#if C_NOISE_GENERATOR_TYPE == 0
    out[idx] = tileable_noise(fx, fy, period, period, seed);
#elif C_NOISE_GENERATOR_TYPE == 1
    out[idx] = gradient_noise(fx, fy, period, period, seed);
#elif C_NOISE_GENERATOR_TYPE == 2
    out[idx] = warped_noise(fx, fy, period, seed);
#endif
}

#define C_NOISE_GENERATOR_STREAM

#ifndef C_NOISE_GENERATOR_STREAM

void CNoiseGenerator::fill(float *host_data, int width, int height) {

    float scale = (float)period / width; // Assuming square images (also period must be an integer for seamless)

    size_t size = width * height * sizeof(float);
    float *dev_data = nullptr;
    cudaMalloc(&dev_data, size);
    cudaMemcpy(dev_data, host_data, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    generate_noise<<<gridDim, blockDim>>>(dev_data, width, height, scale, period, seed);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(host_data, dev_data, size, cudaMemcpyDeviceToHost); // copy back
    cudaFree(dev_data);                                            // free
}

#else

void CNoiseGenerator::fill(float *host_data, int width, int height) {
    float scale = static_cast<float>(period) / width;

    size_t size = width * height * sizeof(float);
    float *dev_data = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&dev_data, size);
    cudaMemcpyAsync(dev_data, host_data, size, cudaMemcpyHostToDevice, stream);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    generate_noise<<<gridDim, blockDim, 0, stream>>>(dev_data, width, height, scale, period, seed);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        // std::cout << "CUDA kernel launch failed!\n";
    }

    cudaMemcpyAsync(host_data, dev_data, size, cudaMemcpyDeviceToHost, stream);

    // Wait for all operations in this stream to complete
    cudaStreamSynchronize(stream);

    cudaFree(dev_data);
    cudaStreamDestroy(stream);
}

#endif

} // namespace c_noise_generator