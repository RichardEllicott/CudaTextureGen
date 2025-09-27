// // src/erosion.cu
// #include "erosion.h"
// #include <cuda_runtime.h>
// #include <algorithm>

#include "erosion.h"
#include <cuda_runtime.h>

// __global__ void rain_kernel(float *water, int width, int height, float rain_rate)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x < width && y < height)
//     {
//         int idx = y * width + x;
//         water[idx] += rain_rate;
//     }
// }

// __global__ void flow_kernel(float *height, float *water, float *outflow, int width, int height)
// {
//     // For each cell, compute flow to 4 neighbors
//     // Store outflow in temporary buffer
// }

// __global__ void erosion_kernel(float *height, float *water, float *sediment, int width, int height, float erosion_rate)
// {
//     // Erode terrain based on water velocity and sediment capacity
// }

// __global__ void evaporation_kernel(float *water, int width, int height, float evap_rate)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x < width && y < height)
//     {
//         int idx = y * width + x;
//         water[idx] *= (1.0f - evap_rate);
//     }
// }

// host side loop
// for (int s = 0; s < steps; ++s) {
//     rain_kernel<<<grid, block>>>(dev_water, width, height, rain_rate);
//     flow_kernel<<<grid, block>>>(dev_height, dev_water, dev_outflow, width, height);
//     erosion_kernel<<<grid, block>>>(dev_height, dev_water, dev_sediment, width, height, erosion_rate);
//     evaporation_kernel<<<grid, block>>>(dev_water, width, height, evap_rate);
// }


// #pragma region PARMATERS

// __constant__ float rain_rate;
// __constant__ float evap_rate;
// __constant__ float erosion_rate;

// struct ErosionParams
// {
//     float rain_rate;
//     float evap_rate;
//     float erosion_rate;
// };

// __constant__ ErosionParams sim_params;

// void setup()
// {
//     float r = 1.0f, e = 0.05f, d = 0.1f;
//     cudaMemcpyToSymbol(rain_rate, &r, sizeof(float));
//     cudaMemcpyToSymbol(evap_rate, &e, sizeof(float));
//     cudaMemcpyToSymbol(erosion_rate, &d, sizeof(float));

//     ErosionParams params = {1.0f, 0.05f, 0.1f};
//     cudaMemcpyToSymbol(sim_params, &params, sizeof(ErosionParams));
// }

// #pragma endregion




__global__ void erosion_kernel(float *data, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int idx = y * width + x;
        data[idx] = fmaxf(0.0f, data[idx] - 1.0f);
    }
}

void run_erosion(float *host_data, int width, int height, int steps)
{
    size_t size = width * height * sizeof(float);
    float *dev_data = nullptr;

    // Allocate once
    cudaMalloc(&dev_data, size);
    cudaMemcpy(dev_data, host_data, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Loop on the host, but keep data on device
    for (int s = 0; s < steps; ++s)
    {
        erosion_kernel<<<grid, block>>>(dev_data, width, height);
    }
    cudaDeviceSynchronize();

    // Copy back once
    cudaMemcpy(host_data, dev_data, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_data);
}
