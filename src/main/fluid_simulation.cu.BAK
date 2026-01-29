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

__global__ void update_wave2(
    int map_width, int map_height, Parameters *pars,
    const float *water_map_prev, // previous water heights
    const float *water_map_curr, // current water heights
    float *water_map_next,       // next water heights
    const float *terrain_map     // terrain heightmap (optional)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= map_width || y >= map_height)
        return;

    int idx = y * map_width + x;

    // Current height
    float h = water_map_curr[idx];

    // Neighbor sampling (clamp at edges)
    float hL = (x > 0) ? water_map_curr[idx - 1] : h;
    float hR = (x < map_width - 1) ? water_map_curr[idx + 1] : h;
    float hU = (y > 0) ? water_map_curr[idx - map_width] : h;
    float hD = (y < map_height - 1) ? water_map_curr[idx + map_width] : h;

    // Discrete Laplacian
    float laplacian = (hL + hR + hU + hD - 4.0f * h);

    // BLOWS UP
    // // Discrete Laplacian (second derivative)
    // float laplacian = (hL + hR + hU + hD - 4.0f * h);
    // // Update rule: wave equation approximation
    // float new_h = h + pars->wave_speed * pars->dt * laplacian;

    // Second-order wave update
    float c = pars->wave_speed;
    float dt = pars->dt;
    float new_h = 2.0f * h - water_map_prev[idx] + (c * c) * (dt * dt) * laplacian;

    // Optional damping
    new_h *= (1.0f - pars->damping);

    // Terrain clamp
    if (terrain_map != nullptr) {
        float ground = terrain_map[idx];
        if (new_h < ground)
            new_h = ground;
    }

    water_map_next[idx] = new_h;
}


__global__ void update_wave3(
    int map_width, int map_height, Parameters *pars,
    const float *water_prev,
    const float *water_curr,
    float *water_next,
    const float *terrain_map
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height) return;

    int idx = y * map_width + x;

    float h = water_curr[idx];

    // Neighbors (clamped)
    float hL = (x > 0) ? water_curr[idx - 1]         : h;
    float hR = (x < map_width - 1) ? water_curr[idx + 1] : h;
    float hU = (y > 0) ? water_curr[idx - map_width] : h;
    float hD = (y < map_height - 1) ? water_curr[idx + map_width] : h;

    // Properly scaled Laplacian
    float dx = pars->cell_size;
    float lap = (hL + hR + hU + hD - 4.0f * h) / (dx * dx);

    // Parameters
    float c  = pars->wave_speed;
    float dt = pars->dt;
    float c2dt2 = (c * c) * (dt * dt);

    // Velocity-based damping
    float vel = h - water_prev[idx]; // discrete velocity
    float new_h = 2.0f * h - water_prev[idx] + c2dt2 * lap
                  - pars->damping * vel;

    // Terrain clamp if needed
    if (terrain_map != nullptr) {
        float ground = terrain_map[idx];
        if (new_h < ground) new_h = ground;
    }

    water_next[idx] = new_h;
}



void TEMPLATE_CLASS_NAME::allocate_device() {

    // if water map empty, fill it with zeros
    if (water_map.empty()) {
        water_map.resize(pars.width, pars.height);
        water_map.zero_device();
    }
    // ensure pars match dimensions
    pars.width = water_map.width();
    pars.height = water_map.height();

    // resize second array to match (this array will be written to, uninitialized memory no problem)
    water_map_next.resize(pars.width, pars.height);

    // if heightmap not empty but dimensions wrong, free this array to avoid errors
    if (!height_map.empty()) {
        if (height_map.width() != pars.width || height_map.height() != pars.height) {
            height_map.free_device();
        }
    }

    if (pars.mode == 1) {
        // if water_map_previous dimensions not the same or even empty
        if (water_map_previous.width() != pars.width || water_map_previous.height() != pars.height) {
            water_map_previous = water_map;
        }
    }
}

void TEMPLATE_CLASS_NAME::deallocate_device() {

    water_map.free_device();
    water_map_next.free_device();
    water_map_previous.free_device();
}

void TEMPLATE_CLASS_NAME::process() {

    allocate_device();

    sync_pars();

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    if (pars.mode == 0) {

        for (int i = 0; i < pars.steps; i++) {

            update_wave<<<grid, block, 0, stream.get()>>>(
                pars.width, pars.height, dev_pars.dev_ptr(),
                water_map.dev_ptr(), water_map_next.dev_ptr(),
                height_map.dev_ptr());

            std::swap(water_map, water_map_next);
        }

    } else if (pars.mode == 1) {

        update_wave3<<<grid, block, 0, stream.get()>>>(
            pars.width, pars.height, dev_pars.dev_ptr(),
            water_map_previous.dev_ptr(),
            water_map.dev_ptr(),
            water_map_next.dev_ptr(),
            height_map.dev_ptr());

        // Rotate buffers (pointer swap, no copy)
        water_map_previous.swap(water_map);
        water_map.swap(water_map_next);
    }

    stream.sync();

    // deallocate_device();
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
