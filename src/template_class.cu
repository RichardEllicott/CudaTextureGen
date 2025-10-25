#include "template_class.cuh"
// #include <assert.h> // optional

namespace template_class {

// a kernel example
__global__ void process_texture(const Parameters *pars, Maps *maps, const size_t width, const size_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    float blend = 0.0f;
    if (maps->blend_mask) {
        blend = maps->blend_mask[idx]; // Safe to access
    }

    // optional
    // assert(maps != nullptr);
    // assert(maps->height_map != nullptr);
    // assert(idx >= 0 && idx < width * height);

    if (maps->height_map) {
        maps->height_map[idx] += blend;

        const int tile_size = 16; // probabally will be inlined (could also use constexpr)

        if ((x / tile_size + y / tile_size) % 2 == 0) { // for test make into a chequer pattern
            maps->height_map[idx] = 0.0;
        }
    }
}

void TemplateClass::process() {

    allocate_and_copy_to_gpu();

    dim3 block(16, 16);
    dim3 grid((host_pars.width + block.x - 1) / block.x,
              (host_pars.height + block.y - 1) / block.y);

    // Launch kernel with access to private members
    process_texture<<<grid, block>>>(device_pars, device_map_struct, host_pars.width, host_pars.height);

    CUDA_CHECK(cudaGetLastError());      // Check launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Check execution

    copy_maps_back_from_gpu();

    free_memory();
}

} // namespace template_class
