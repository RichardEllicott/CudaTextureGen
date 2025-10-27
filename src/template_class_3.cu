#include "template_class_3.cuh"

namespace TEMPLATE_NAMESPACE {

// a kernel example makes a chequer pattern
__global__ void process_texture(const Parameters *pars, float *height_map, const size_t width, const size_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    if (height_map) {
        const int tile_size = 16;                       // probabally will be inlined (could also use constexpr)
        if ((x / tile_size + y / tile_size) % 2 == 0) { // for test make into a chequer pattern
            height_map[idx] = 0.0;
        }
    }
}

void TEMPLATE_CLASS_NAME::process() {

    image.upload_to_device();

    pars.width = image.get_width();
    pars.height = image.get_height();

    // #define X(TYPE, NAME) \
    //     NAME.upload_to_device();
    //     TEMPLATE_CLASS_MAPS
    // #undef X

    core::CudaStruct<Parameters> gpu_pars(pars); // automaticly uploads and free

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    process_texture<<<grid, block>>>(gpu_pars.device_ptr, image.device_ptr, pars.width, pars.height);

    image.download_from_device();
    image.free_device_memory();

    // #define X(TYPE, NAME)            \
    //     NAME.download_from_device(); \
    //     NAME.free_device_memory();   \
    //     TEMPLATE_CLASS_MAPS
    // #undef X
}

} // namespace TEMPLATE_NAMESPACE
