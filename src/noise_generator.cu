/*

started with co-pilot... solved very quick with claude:


https://claude.ai/chat/626e97e1-b6cb-4690-9dc7-db32b31ccccf


*/
#include "noise_generator.cuh"

namespace noise_generator {
using namespace noise_util;

#pragma region NOISE

__device__ float value_noise(float x, float y, int period_x, int period_y, int seed) {
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

    float a = hash_scalar(xi0, yi0, seed);
    float b = hash_scalar(xi1, yi0, seed);
    float c = hash_scalar(xi0, yi1, seed);
    float d = hash_scalar(xi1, yi1, seed);

    float x1 = lerp(a, b, u);
    float x2 = lerp(c, d, u);

    return lerp(x1, x2, v);
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

    return lerp(x1, x2, v); // Remap to [0,1]
}

// __device__ float warped_noise(float x, float y, int period, int seed) {
//     // Sample noise at offset positions
//     float warp_x = value_noise(x + 5.2f, y + 1.3f, period, period, seed + 1) * 4.0f;
//     float warp_y = value_noise(x + 3.7f, y + 9.1f, period, period, seed + 2) * 4.0f;

//     // Use warped coordinates for final sample
//     return value_noise(x + warp_x, y + warp_y, period, period, seed);
// }

// exposing parameters
__device__ float warped_noise(
    float x, float y,
    int period_x, int period_y,
    int seed,
    float warp_amp,  // how strong the warp is
    float warp_scale // frequency of the warp field
) {
    float warp_x = value_noise(x * warp_scale + 5.2f,
                               y * warp_scale + 1.3f,
                               period_x, period_y,
                               seed + 1) *
                   warp_amp;

    float warp_y = value_noise(x * warp_scale + 3.7f,
                               y * warp_scale + 9.1f,
                               period_x, period_y,
                               seed + 2) *
                   warp_amp;

    return value_noise(x + warp_x, y + warp_y, period_x, period_y, seed);
}

__device__ float value_noise(const float x, const float y, const float z,
                             const int period_x, const int period_y, const int period_z,
                             const int seed) {

    int xi = floorf(x), yi = floorf(y), zi = floorf(z);
    float xf = x - xi, yf = y - yi, zf = z - zi;

    float u = fade(xf), v = fade(yf), w = fade(zf);

    int xi0 = posmod(xi, period_x), xi1 = posmod(xi + 1, period_x);
    int yi0 = posmod(yi, period_y), yi1 = posmod(yi + 1, period_y);
    int zi0 = posmod(zi, period_z), zi1 = posmod(zi + 1, period_z);

    float c000 = hash_scalar(xi0, yi0, zi0, seed);
    float c100 = hash_scalar(xi1, yi0, zi0, seed);
    float c010 = hash_scalar(xi0, yi1, zi0, seed);
    float c110 = hash_scalar(xi1, yi1, zi0, seed);
    float c001 = hash_scalar(xi0, yi0, zi1, seed);
    float c101 = hash_scalar(xi1, yi0, zi1, seed);
    float c011 = hash_scalar(xi0, yi1, zi1, seed);
    float c111 = hash_scalar(xi1, yi1, zi1, seed);

    float x00 = lerp(c000, c100, u);
    float x10 = lerp(c010, c110, u);
    float x01 = lerp(c001, c101, u);
    float x11 = lerp(c011, c111, u);

    float y0 = lerp(x00, x10, v);
    float y1 = lerp(x01, x11, v);

    return lerp(y0, y1, w);
}

__device__ float gradient_noise(float x, float y, float z,
                                int period_x, int period_y, int period_z,
                                int seed) {
    int xi = (int)floorf(x);
    int yi = (int)floorf(y);
    int zi = (int)floorf(z);

    float xf = x - xi;
    float yf = y - yi;
    float zf = z - zi;

    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);

    int xi0 = posmod(xi, period_x);
    int yi0 = posmod(yi, period_y);
    int zi0 = posmod(zi, period_z);
    int xi1 = posmod(xi + 1, period_x);
    int yi1 = posmod(yi + 1, period_y);
    int zi1 = posmod(zi + 1, period_z);

    // Get gradients at cube corners
    float3 g000 = gradient(xi0, yi0, zi0, seed);
    float3 g100 = gradient(xi1, yi0, zi0, seed);
    float3 g010 = gradient(xi0, yi1, zi0, seed);
    float3 g110 = gradient(xi1, yi1, zi0, seed);
    float3 g001 = gradient(xi0, yi0, zi1, seed);
    float3 g101 = gradient(xi1, yi0, zi1, seed);
    float3 g011 = gradient(xi0, yi1, zi1, seed);
    float3 g111 = gradient(xi1, yi1, zi1, seed);

    // Distance vectors
    float3 d000 = make_float3(xf, yf, zf);
    float3 d100 = make_float3(xf - 1.0f, yf, zf);
    float3 d010 = make_float3(xf, yf - 1.0f, zf);
    float3 d110 = make_float3(xf - 1.0f, yf - 1.0f, zf);
    float3 d001 = make_float3(xf, yf, zf - 1.0f);
    float3 d101 = make_float3(xf - 1.0f, yf, zf - 1.0f);
    float3 d011 = make_float3(xf, yf - 1.0f, zf - 1.0f);
    float3 d111 = make_float3(xf - 1.0f, yf - 1.0f, zf - 1.0f);

    // Dot products
    float v000 = dot(g000, d000);
    float v100 = dot(g100, d100);
    float v010 = dot(g010, d010);
    float v110 = dot(g110, d110);
    float v001 = dot(g001, d001);
    float v101 = dot(g101, d101);
    float v011 = dot(g011, d011);
    float v111 = dot(g111, d111);

    // Interpolate
    float x00 = lerp(v000, v100, u);
    float x10 = lerp(v010, v110, u);
    float x01 = lerp(v001, v101, u);
    float x11 = lerp(v011, v111, u);

    float y0 = lerp(x00, x10, v);
    float y1 = lerp(x01, x11, v);

    return lerp(y0, y1, w);
}

// hash noise test
__device__ float hash_noise(int x, int y, int seed) {
    return hash_scalar(x, y, seed);
}

// hash noise test
__device__ float hash_noise(int x, int y, int z, int seed) {
    return hash_scalar(x, y, z, seed);
}

// 🚧
// float fbm(float x, float y, float z, int seed, int octaves = 5) {
//     float value = 0.0f;
//     float amplitude = 0.5f;
//     float frequency = 1.0f;

//     for (int i = 0; i < octaves; ++i) {
//         value += amplitude * gradient_noise(x * frequency, y * frequency, z * frequency, 64, 64, 64, seed + i);
//         frequency *= 2.0f;
//         amplitude *= 0.5f;
//     }

//     return value;
// }

// voronoi, no wrap yet
__global__ void voronoi_kernel(float *output, int width, int height,
                               float2 *feature_points, int num_points) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    float min_dist = 1e20f;

    float2 p = make_float2(x, y);

    for (int i = 0; i < num_points; ++i) {
        float2 fp = feature_points[i];
        float dx = p.x - fp.x;
        float dy = p.y - fp.y;
        float dist = dx * dx + dy * dy; // squared distance
        if (dist < min_dist)
            min_dist = dist;
    }

    output[idx] = sqrtf(min_dist); // or leave squared for performance
}

// USAGE
//
// dim3 block(16, 16);
// dim3 grid((width + block.x - 1) / block.x,
//           (height + block.y - 1) / block.y);
// voronoi_kernel<<<grid, block>>>(output, width, height, d_feature_points, num_points);
//
//

#pragma endregion

#pragma region GLOBAL

__global__ void generate_noise(float *out, const unsigned width, const unsigned height,
                               const Parameters *const __restrict__ pars) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const int idx = y * width + x;

    // Key fix: scale determines the noise frequency
    // The period should match the scaled coordinate space
    const float fx = x * pars->scale + pars->x;
    const float fy = y * pars->scale + pars->y;

    switch (pars->type) {

    case 0: // gradient noise 2D, like simplex rounded and smooth
        out[idx] = gradient_noise(
            fx, fy,
            pars->period, pars->period,
            pars->seed);
        break;
    case 1: // value noise 2D (very blocky)
        out[idx] = value_noise(
            fx, fy,
            pars->period, pars->period,
            pars->seed);
        break;
    case 2: // warped value noise 2D
        out[idx] = warped_noise(
            fx, fy,
            pars->period, pars->period, pars->seed, pars->warp_amp, pars->warp_scale);
        break;
    case 3: // value noise 3D
        out[idx] = value_noise(
            fx, fy, pars->z,
            pars->period, pars->period, pars->period,
            pars->seed);
        break;
    case 4: // gradient noise 3D
        out[idx] = gradient_noise(
            fx, fy, pars->z,
            pars->period, pars->period, pars->period,
            pars->seed);
        break;

    case 5: // 2D hash test
        out[idx] = hash_noise(x, y, pars->seed);
        break;
    case 6: // 3D hash test
        out[idx] = hash_noise(x, y, pars->z, pars->seed);
        break;
    }
}

void NoiseGenerator::fill(float *host_data, const unsigned width, const unsigned height) {

    pars.scale = static_cast<float>(pars.period) / width;

    size_t size = width * height * sizeof(float);
    float *dev_data = nullptr;

    cudaStream_t stream; // optional, make a stream for this
    cudaStreamCreate(&stream);

    Parameters *dev_pars;
    cudaMalloc(&dev_pars, sizeof(Parameters));
    cudaMemcpy(dev_pars, &pars, sizeof(Parameters), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_data, size);
    cudaMemcpyAsync(dev_data, host_data, size, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    generate_noise<<<grid, block, 0, stream>>>(dev_data, width, height, dev_pars);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpyAsync(host_data, dev_data, size, cudaMemcpyDeviceToHost, stream);

    // Wait for all operations in this stream to complete
    cudaStreamSynchronize(stream);

    cudaFree(dev_data);
    cudaFree(dev_pars);
    cudaStreamDestroy(stream);
}

#pragma endregion

} // namespace noise_generator