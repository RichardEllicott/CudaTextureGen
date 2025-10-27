#include "template_class2.cuh"

namespace TEMPLATE_CLASS_NAMESPACE {

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

void TEMPLATE_CLASS_NAME::process() {

    // ⚠️ ensuring the map sizes match
    pars.width = height_map.get_width();
    pars.height = height_map.get_height();

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    allocate_device_memory();
    clear_maps_on_device(); // optional clear maps
    copy_maps_to_device();

    process_texture<<<grid, block>>>(dev_pars, dev_maps, pars.width, pars.height);

    copy_maps_from_device();
    free_device_memory();
}

} // namespace TEMPLATE_CLASS_NAMESPACE