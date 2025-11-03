#include "erosion4.cuh"

namespace TEMPLATE_NAMESPACE {

#include <curand_kernel.h>


__global__ void simple_erode(
    const int width, const int height,
    const float *heightmap, const float *sediment,
    float *heightmap_out, float *sediment_out,
    curandState *rand_states = nullptr,
    const bool wrap = true,
    const float jitter = 0.0f,
    const float erosion_rate = 0.01f,
    const float slope_threshold = 0.01f,
    const float deposition_rate = 0.01f
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float h = heightmap[idx];
    float s = sediment[idx];

    // 8-way neighbor offsets
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    float total_slope = 0.0f;
    float slopes[8] = {0};

    // Compute slopes
    for (int i = 0; i < 8; ++i) {
        int nx, ny;
        if (wrap) {
            nx = (x + dx[i] + width) % width;
            ny = (y + dy[i] + height) % height;
        } else {
            nx = x + dx[i];
            ny = y + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        }

        int nIdx = ny * width + nx;
        float nh = heightmap[nIdx];
        float slope = h - nh;

        if (rand_states && jitter > 0.0f) {
            float rand = curand_uniform(&rand_states[idx]);
            slope += rand * jitter;
        }

        if (slope > slope_threshold) {
            slopes[i] = slope;
            total_slope += slope;
        }
    }

    // Local erosion
    float eroded = erosion_rate * total_slope;
    h -= eroded;
    s += eroded;

    // Distribute sediment to neighbors (into OUT buffers)
    for (int i = 0; i < 8; ++i) {
        if (slopes[i] > 0) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            int nIdx = ny * width + nx;
            float share = (slopes[i] / total_slope) * deposition_rate * s;

            atomicAdd(&heightmap_out[nIdx], share);
            atomicAdd(&sediment_out[nIdx], -share);
        }
    }

    // Write back this cell’s updated values into OUT buffers
    atomicAdd(&heightmap_out[idx], h);
    atomicAdd(&sediment_out[idx], s);
}


void TEMPLATE_CLASS_NAME::process() {

    height_map.upload();

    pars.width = height_map.get_width();
    pars.height = height_map.get_height();

    sediment_map.resize(pars.width, pars.height);
    sediment_map.clear();
    sediment_map.upload();



    // core::CudaArrayManager


    core::cuda::Stream stream;

    core::cuda::Struct<Parameters> _pars(pars); // automaticly uploads and free

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    // process_texture<<<grid, block, 0, stream.get()>>>(
    //     _pars.dev_ptr(), image.dev_ptr(),
    //     pars.width, pars.height);

    stream.sync();

    height_map.download();
    height_map.free_device();
}

} // namespace TEMPLATE_NAMESPACE
