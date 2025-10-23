#include "template_class.h"

namespace template_class {



// a kernel example
__global__ void process_texture(Parameters *pars, Maps *maps, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    float blend = 0.0f;
    if (maps->blend_mask) {
        blend = maps->blend_mask[idx]; // Safe to access
    }

    // // Use blend in computation...
}

void TemplateClass::process() {

    allocate_memory();

    dim3 block(16, 16);
    dim3 grid((host_pars.width + block.x - 1) / block.x,
              (host_pars.height + block.y - 1) / block.y);

    // Launch kernel with access to private members
    process_texture<<<block, grid>>>(device_pars, device_map_struct, host_pars.width, host_pars.height);
    CUDA_CHECK(cudaGetLastError());      // Check launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Check execution

    free_memory();
}

} // namespace template_class