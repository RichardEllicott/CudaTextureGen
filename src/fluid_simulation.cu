#include "fluid_simulation.cuh"

namespace TEMPLATE_NAMESPACE {

// __global__ void updateVelocityKernel(
//     Parameters *pars,
//     const int width, const int height,
//     const float *height_map_in,
//     const float *height_map_out,
//     const float2 *velocity_map_in,
//     float2 *velocity_map_in_out
//     ) {

//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x >= width || y >= height)
//         return;
// }

void TEMPLATE_CLASS_NAME::allocate_device() {

    // heightmap
    if (height_map.empty()) {
        height_map.resize(pars.width, pars.height);
    }
    pars.width = height_map.get_width();
    pars.height = height_map.get_height();
    height_map.upload();

    // heightmap_out (match the same size)
    height_map_out.resize(pars.width, pars.height);
    height_map_out.allocate_device();

    // velocity map setup ping pong
    velocity_map.resize(pars.width, pars.height);
    velocity_map.clear();
    velocity_map.allocate_device();
    velocity_map_out.resize(pars.width, pars.height);
    velocity_map.clear();
    velocity_map_out.allocate_device();
}
void TEMPLATE_CLASS_NAME::deallocate_device() {

    height_map.download();
    height_map.free_device();
    height_map_out.download();
    height_map_out.free_device();

    velocity_map.free_device();
    velocity_map_out.free_device();
}

void TEMPLATE_CLASS_NAME::process() {

    allocate_device();

    core::cuda::Stream stream;

    core::CudaStruct<Parameters> _pars(pars); // automaticly uploads and free

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);
    //
    //
    float *_height_map = height_map.dev_ptr();         // ping pong pointers
    float *_height_map_out = height_map_out.dev_ptr(); // ping pong pointers

    float *_velocity_map = velocity_map.dev_ptr();         // ping pong pointers
    float *_velocity_map_out = velocity_map_out.dev_ptr(); // ping pong pointers

    for (int i = 0; i < pars.steps; i++) {

        //
        // process_texture<<<grid, block>>>(_pars.dev_ptr(), _map_ptrs.dev_ptr(), pars.width, pars.height);

        // advectKernel<<<grid, block, 0, stream.get()>>>()

        //     pars.width, pars.height, _pars.dev_ptr(),
        //     _height_map, _height_map_out,
        //     nullptr
        // );

        std::swap(_height_map, _height_map_out);
        std::swap(_velocity_map, _velocity_map_out);
    }
    //
    //
    //

    stream.sync();

    deallocate_device();
}

} // namespace TEMPLATE_NAMESPACE
