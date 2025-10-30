#include "tectonics.cuh"

namespace TEMPLATE_NAMESPACE {

__device__ bool shouldFracture(float stress, float threshold) {
    return stress > threshold;
}

__global__ void propagate_pressure(float *pressure, float *velocity, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    // Simple wave equation update
    float laplacian = pressure[idx - 1] + pressure[idx + 1] + pressure[idx - width] + pressure[idx + width] - 4 * pressure[idx];
    velocity[idx] += laplacian * 0.01f; // stiffness factor
    pressure[idx] += velocity[idx];
}

void TEMPLATE_CLASS_NAME::process() {

    height_map.upload();

    pars.width = height_map.get_width();
    pars.height = height_map.get_height();

    // #define X(TYPE, NAME) \
    //     NAME.upload_to_device();
    //     TEMPLATE_CLASS_MAPS
    // #undef X

    core::CudaStruct<Parameters> gpu_pars(pars); // automaticly uploads and free

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    // process_texture<<<grid, block>>>(gpu_pars.dev_ptr(), image.dev_ptr(), pars.width, pars.height);

    height_map.download();
    height_map.free_device();

    // #define X(TYPE, NAME)            \
    //     NAME.download_from_device(); \
    //     NAME.free_device_memory();   \
    //     TEMPLATE_CLASS_MAPS
    // #undef X
}

} // namespace TEMPLATE_NAMESPACE
