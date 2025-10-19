#include "resample.cuh"

namespace resample {

__device__ __forceinline__ int wrap_coord(float v, float max) {
    float m = fmodf(v, max);
    return (m < 0.0f) ? m + max : m;
}

__device__ __forceinline__ int clamp_coord(float v, float max) {
    return fminf(fmaxf(v, 0.0f), (max - 1));
}

__device__ float sample_bilinear(const float *img,
                                 int width, int height,
                                 float x, float y,
                                 bool wrap) {

    // Apply addressing mode
    if (wrap) {
        x = wrap_coord(x, width);
        y = wrap_coord(y, height);
    } else {
        x = clamp_coord(x, width);
        y = clamp_coord(y, height);
    }

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = (x0 + 1) % width;
    int y1 = (y0 + 1) % height;

    float dx = x - x0;
    float dy = y - y0;

    float v00 = img[y0 * width + x0];
    float v10 = img[y0 * width + x1];
    float v01 = img[y1 * width + x0];
    float v11 = img[y1 * width + x1];

    return (1 - dx) * (1 - dy) * v00 +
           dx * (1 - dy) * v10 +
           (1 - dx) * dy * v01 +
           dx * dy * v11;
}

__global__ void resample_kernel(const float *input, float *output,
                                int in_w, int in_h,
                                int out_w, int out_h,
                                const float *map_x, const float *map_y) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h)
        return;

    int idx = y * out_w + x;

    // map_x/map_y give the input coords for this output pixel
    float src_x = map_x[idx];
    float src_y = map_y[idx];

    output[idx] = sample_bilinear(input, in_w, in_h, src_x, src_y, true);
}

void Resample::process_maps(
    const float *host_in, float *host_out,
    const int width, const int height,
    const float *map_x, const float *map_y) {

    size_t size = width * height * sizeof(float);

    float *dev_in = nullptr;
    float *dev_out = nullptr;
    float *dev_map_x = nullptr;
    float *dev_map_y = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate device buffers
    cudaMalloc(&dev_in, size);
    cudaMalloc(&dev_out, size);
    cudaMalloc(&dev_map_x, size);
    cudaMalloc(&dev_map_y, size);

    // Copy input + maps
    cudaMemcpyAsync(dev_in, host_in, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_map_x, map_x, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_map_y, map_y, size, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Launch resample kernel
    resample_kernel<<<grid, block, 0, stream>>>(dev_in, dev_out,
                                                width, height,
                                                width, height,
                                                dev_map_x, dev_map_y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Copy result back
    cudaMemcpyAsync(host_out, dev_out, size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_map_x);
    cudaFree(dev_map_y);
    cudaStreamDestroy(stream);
}

} // namespace resample