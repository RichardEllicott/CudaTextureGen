#include "fluid_simulation.cuh"

namespace TEMPLATE_NAMESPACE {

// Simple CUDA kernel for wave propagation
__global__ void update_wave(
    int map_width, int map_height, Parameters *pars,
    const float *water_map_in, // current water heights
    float *water_map_out,      // next water heights
    const float *terrain_map   // terrain heightmap (optional, can be nullptr)

) {
    // Compute 2D thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= map_width || y >= map_height)
        return;

    int idx = y * map_width + x;

    // Read current height
    float h = water_map_in[idx];

    // Neighbor sampling (clamp at edges)
    float hL = (x > 0) ? water_map_in[idx - 1] : h;
    float hR = (x < map_width - 1) ? water_map_in[idx + 1] : h;
    float hU = (y > 0) ? water_map_in[idx - map_width] : h;
    float hD = (y < map_height - 1) ? water_map_in[idx + map_width] : h;

    // Discrete Laplacian (second derivative)
    float laplacian = (hL + hR + hU + hD - 4.0f * h);

    // Update rule: wave equation approximation
    float new_h = h + pars->wave_speed * pars->dt * laplacian;

    // Terrain clamp (if terrain provided)
    if (terrain_map != nullptr) {
        float ground = terrain_map[idx];
        if (new_h < ground)
            new_h = ground; // no water below terrain
    }

    water_map_out[idx] = new_h;
}

void TEMPLATE_CLASS_NAME::allocate_device() {

    // heightmap
    if (water_map.empty()) {
        water_map.resize(pars.width, pars.height);
        water_map.clear(); // set as 0
    } else {
        pars.width = water_map.get_width();
        pars.height = water_map.get_height();
    }
    water_map.upload();

    // heightmap_out (match the same size)
    water_map_out.resize(pars.width * pars.height);
    water_map_out.zero_device();


}
void TEMPLATE_CLASS_NAME::deallocate_device() {

    water_map.download();

    water_map.free_device();
    water_map_out.free_device();
}

void TEMPLATE_CLASS_NAME::process() {

    allocate_device();

    core::cuda::Stream stream;

    // core::cuda::DeviceStruct<Parameters> _pars(pars); // automaticly uploads and free

    



    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    // ping pong pointers
    float *_water_map = water_map.dev_ptr();
    float *_water_map_out = water_map_out.dev_ptr();

    for (int i = 0; i < pars.frame_count; i++) {

        update_wave<<<grid, block, 0, stream.get()>>>(
            pars.width, pars.height, dev_pars.dev_ptr(),
            _water_map, _water_map_out,
            nullptr);

        std::swap(_water_map, _water_map_out);
    }


    stream.sync();

    deallocate_device();
}

} // namespace TEMPLATE_NAMESPACE
