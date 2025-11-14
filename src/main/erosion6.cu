/*
Notes

Hydraulic erosion: water carries sediment downhill, depositing when flow slows

Aeolian erosion: wind lifts particles, moves them across terrain, and deposits them when wind speed decreases or slope traps them.

Slope stability Sand tends to accumulate until the slope exceeds the angle of repose (~30–35° for dry sand).
If exceeded, grains slide downhill until equilibrium is restored.


*/

#include "core.h"
#include "erosion6.cuh"
#include "noise_util.cuh"
#include <chrono>
#include <cmath>

namespace TEMPLATE_NAMESPACE {

#pragma region HELPERS

constexpr float SQRT2 = 1.4142135623730950488f;

// 8 offsets with the opposites in pairs
__device__ __constant__ int2 offsets[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
// __device__ __constant__ int2 reverse_offsets[8] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1}};
__device__ __constant__ float offset_distances[8] = {1.0f, 1.0f, 1.0f, 1.0f, SQRT2, SQRT2, SQRT2, SQRT2};
__device__ __constant__ int opposite_offset_refs[8] = {1, 0, 3, 2, 5, 4, 7, 6};

// tested posmod for wrapping map coordinates
__device__ __forceinline__ int posmod(int i, int mod) {
    int result = i % mod;
    return result < 0 ? result + mod : result;
}

// wrap or clamp for map coordinates, note the clamp is range-1
__device__ __forceinline__ int wrap_or_clamp(int i, int range, bool wrap) {
    if (wrap) {
        return posmod(i, range);
    } else {
        return i < 0 ? 0 : (i >= range ? range - 1 : i);
    }
}

__device__ __forceinline__ int clampi(int value, int minimum, int maximum) {
    return min(max(value, minimum), maximum);
}

__device__ __forceinline__ float clampf(float value, float minimum, float maximum) {
    return fminf(fmaxf(value, minimum), maximum);
}

#pragma endregion

#pragma region KERNELS

__global__ void rain_pass(
    const Parameters *pars,
    const int map_width, const int map_height,
    curandState *rand_states,

    const float *__restrict__ rain_map, // 🚧 optional in

    float *__restrict__ water_map // out

) {
    // ================================================================
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height)
        return;
    int idx = y * map_width + x;
    // ================================================================

    // generate a random number r
    curandState local = rand_states[idx];
    float r = curand_uniform(&local); // (0,1]
    rand_states[idx] = local;         // save updated state

    float rain = pars->rain_rate * r;
    water_map[idx] += rain;
}

__global__ void calculate_flux(
    const Parameters *pars,
    const int map_width, const int map_height,
    const int step,

    const float *__restrict__ height_map,   // in
    const float *__restrict__ water_map,    // in
    const float *__restrict__ sediment_map, // in

    float *__restrict__ flux8,          // out
    float *__restrict__ sediment_flux8, // out
    float *__restrict__ slope_map       // out
) {
    // ================================================================
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height)
        return;
    int idx = y * map_width + x;
    // ================================================================

    float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];
    float surface = height + water;

    // ================================================================
    // [Calculate Slopes]
    // ----------------------------------------------------------------
    float slopes[8];        // positive slope to neighbours
    float sum_slope = 0.0f; // total slope

    for (int n = 0; n < 8; ++n) {

        // new pos
        int nx = wrap_or_clamp(x + offsets[n].x, map_width, pars->wrap);
        int ny = wrap_or_clamp(y + offsets[n].y, map_height, pars->wrap);
        int nidx = ny * map_width + nx;

        float n_surface = height_map[nidx] + water_map[nidx]; // new surface
        float difference = surface - n_surface;               // positive means we are higher

        if (pars->slope_jitter > 0.0f) {
            float jitter = pars->slope_jitter * noise_util::trig_hash(x, y, n + 1234, step);
            difference += jitter;
        }

        if (pars->correct_diagonal_distance) {
            difference /= offset_distances[n];
        }

        // amount higher than we are than neighbour (but 0.0 if difference is below slope_threshold)
        float positive_difference = difference > pars->slope_threshold ? difference : 0.0f; // amount higher we are than neighbour (or 0 if we are lower)

        slopes[n] = positive_difference;
        sum_slope += positive_difference;
    }

    slope_map[idx] = sum_slope; // OUT

    // ================================================================
    // [Calculate Flux]
    // ----------------------------------------------------------------
    float max_outflow = fminf(water, pars->max_water_outflow); // slows outflow if above threshold
    float *cell_flux = &flux8[idx * 8];                        // pointer to this cell’s 8‑flux slice

    if (sum_slope > 1e-6f && max_outflow > 0.f) {
        for (int n = 0; n < 8; ++n) {
            float q = (slopes[n] / sum_slope) * max_outflow; // proportional share of outflow
            cell_flux[n] = q;
        }
    } else {
        for (int n = 0; n < 8; ++n) {
            cell_flux[n] = 0.f; // no downhill slope or no water: zero flux
        }
    }

    // ================================================================
    // [Isotropic Diffusion]
    // ----------------------------------------------------------------
    if (pars->diffusion_rate > 0.0f) {

        for (int n = 0; n < 8; ++n) {
            int nx = wrap_or_clamp(x + offsets[n].x, map_width, pars->wrap);
            int ny = wrap_or_clamp(y + offsets[n].y, map_height, pars->wrap);
            int nidx = ny * map_width + nx;

            float neighbor_water = water_map[nidx];
            float delta = neighbor_water - water; // positive if neighbor has more
            float diffusion_flux = pars->diffusion_rate * delta;

            cell_flux[n] += diffusion_flux;
        }
    }

    // ================================================================
    // [Sediment Flux]
    // ----------------------------------------------------------------
    float sediment_concentration = (water > 1e-6f) ? (sediment / water) : 0.0f;
    for (int n = 0; n < 8; ++n) {
        float q = flux8[idx * 8 + n];                             // water flux to neighbor
        sediment_flux8[idx * 8 + n] = sediment_concentration * q; // sediment carried with that water
    }
}

__global__ void apply_flux(
    const Parameters *pars,
    const int map_width, const int map_height,

    const float *__restrict__ height_map,     // in
    const float *__restrict__ water_map,      // in
    const float *__restrict__ sediment_map,   // in
    const float *__restrict__ flux8,          // in
    const float *__restrict__ sediment_flux8, // in
    const float *__restrict__ slope_map,      // in

    const float *__restrict__ hardness_map, // 🚧 optional in

    float *__restrict__ height_map_out,  // out
    float *__restrict__ water_map_out,   // out
    float *__restrict__ sediment_map_out // out

) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height)
        return;
    int idx = y * map_width + x;
    // values change in this pass and are written out
    float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];
    float slope_factor = slope_map[idx]; // the strength of the slope from previous pass

    // ================================================================
    // [Calculate and Apply Flow]
    // ----------------------------------------------------------------
    float water_inflow = 0.f;
    float water_outflow = 0.0f;

    for (int n = 0; n < 8; ++n) {
        // calculate outflow
        water_outflow += flux8[idx * 8 + n];
        // calculate inflow
        int nx = wrap_or_clamp(x + offsets[n].x, map_width, pars->wrap);
        int ny = wrap_or_clamp(y + offsets[n].y, map_height, pars->wrap);
        int nidx = ny * map_width + nx;

        int opposite_offset = opposite_offset_refs[n];
        water_inflow += flux8[nidx * 8 + opposite_offset];
    }
    water -= water_outflow;
    water += water_inflow;

    // ================================================================
    // [Carving (basic optional)]
    // ----------------------------------------------------------------
    height -= water_outflow * pars->outflow_carve;

    // ================================================================
    // [Simple Erode (like previous model)]
    // ----------------------------------------------------------------
    height -= slope_factor * pars->simple_erosion_rate;

    // ================================================================
    // [Erosion]
    // ----------------------------------------------------------------
    float erosion = water_outflow * pars->erosion_rate; // proposed erosion

    switch (pars->erosion_mode) {
    case 0: // slope_factor only
        erosion *= slope_factor;
        break;
    case 1: // soft saturate slope_factor
        erosion *= slope_factor / (1.0f + slope_factor);
        break;
    case 2: // exponent based slope_factor
        erosion *= powf(slope_map[idx], pars->slope_exponent);
        break;
    }

    float available_erosion = height - pars->min_height; // limit erosion to available rock above min_height
    erosion = fminf(erosion, fmaxf(0.0f, available_erosion));
    sediment += erosion;
    height -= erosion;

    height = clampf(height, pars->min_height, pars->max_height); // clamp height (but min is already enforced)

    // ================================================================
    // [Evaporation]
    // ----------------------------------------------------------------
    water -= pars->evaporation_rate;

    // ================================================================
    // [Sediment Transport]
    // ----------------------------------------------------------------
    float sediment_change = 0.0f;
    for (int n = 0; n < 8; ++n) {
        int nx = wrap_or_clamp(x + offsets[n].x, map_width, pars->wrap);
        int ny = wrap_or_clamp(y + offsets[n].y, map_height, pars->wrap);
        int nidx = ny * map_width + nx;
        int opp = opposite_offset_refs[n];

        sediment_change -= sediment_flux8[idx * 8 + n];    // outflow
        sediment_change += sediment_flux8[nidx * 8 + opp]; // inflow
    }
    sediment += sediment_change;

    // ================================================================
    // [Deposition]
    // ----------------------------------------------------------------
    if (pars->deposition_mode == 0) {
        // basic deposition
        float deposit = sediment * pars->deposition_rate;
        sediment -= deposit;
        height += deposit;
    } else if (pars->deposition_mode == 1) {
        // capacity based
        float capacity = pars->sediment_capacity * water * slope_factor;
        if (sediment > capacity) {
            float deposit = sediment_map[idx] - capacity;
            sediment -= deposit;
            height += deposit;
        }
    }

    // ================================================================
    height_map_out[idx] = height;
    water_map_out[idx] = fmaxf(0.f, water);       // no negative water
    sediment_map_out[idx] = fmaxf(0.f, sediment); // no negative sediment
}

// very simple with no shared memory
__global__ void debug_pass(
    const Parameters *pars,
    const int map_width, const int map_height,
    const int step,

    const float *__restrict__ height_map,
    const float *__restrict__ water_map,
    const float *__restrict__ sediment_map,

    DebugData *debug_array) {
    // ================================================================
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height)
        return;
    int idx = y * map_width + x;
    // ================================================================
    float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];

    atomicAdd(&debug_array[step].total_height, height);
    atomicAdd(&debug_array[step].total_water, water);
    atomicAdd(&debug_array[step].total_sediment, sediment);

    // float h = height_map[idx];
    // float w = water_map[idx];
    // float s = sediment_map[idx];

    // if (idx == 0) {
    //     printf("step %d idx=%d h=%f w=%f s=%f\n", step, idx, h, w, s);
    // }
}

#pragma endregion

#pragma region MAIN

// allocate memory
void TEMPLATE_CLASS_NAME::allocate_device() {

    if (device_allocated)
        return;

    device_allocated = true;

    height_map.upload();
    pars._width = height_map.get_width();
    pars._height = height_map.get_height();

    if (water_map.size() != height_map.size()) { // if we don't start with a map clear
        water_map.resize(pars._width, pars._height);
        water_map.clear();
    }
    water_map.upload();

    if (sediment_map.size() != height_map.size()) { // if we don't start with a map clear
        sediment_map.resize(pars._width, pars._height);
        sediment_map.clear();
    }
    sediment_map.upload();

    // optional map
    if (hardness_map.get_width() == pars._width && hardness_map.get_height() == pars._height) {
        hardness_map.upload();
    }
    if (rain_map.get_width() == pars._width && rain_map.get_height() == pars._height) {
        rain_map.upload();
    }

    // ================================================================

    size_t array_size = pars._width * pars._height;

    // private device arrays
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    NAME.resize(array_size *DIMENSION);       \
    NAME.zero_device();
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X

    //     cudaDeviceSynchronize();
}

// deallocate memory
void TEMPLATE_CLASS_NAME::deallocate_device() {

    auto timer = core::Timer();

    device_allocated = false;

    // free maps
#define X(TYPE, NAME, DESCRIPTION) \
    NAME.free_device();            \
    TEMPLATE_CLASS_MAPS
#undef X

    // free device arrays
#define X(TYPE, DIM, NAME, DESCRIPTION) \
    NAME.free_device();
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X

    timer.mark_time();
    printf("deallocate device time: %.3f seconds\n", timer.elapsed_seconds());
}

// run erosion
void TEMPLATE_CLASS_NAME::process() {

    printf("<<< Erosion Process >>>\n");

    auto timer = core::Timer();

    allocate_device();
    core::cuda::Stream stream;
    core::cuda::DeviceStruct<Parameters> dev_pars(pars); // automaticly uploads and free

    auto curand_array_2d = core::cuda::CurandArray2D(pars._width, pars._height, stream.get());
    stream.sync(); // important??

    timer.mark_time();
    printf("allocate device time: %.3f seconds\n", timer.elapsed_seconds());

    // pointers for swapping the maps around (ping/pong)
    float *dev_height_map_in = height_map.dev_ptr();
    float *dev_water_map_in = water_map.dev_ptr();
    float *dev_sediment_map_in = sediment_map.dev_ptr();

    float *dev_height_map_out = height_map_out.dev_ptr();
    float *dev_water_map_out = water_map_out.dev_ptr();
    float *dev_sediment_map_out = sediment_map_out.dev_ptr();

    // debug output
    core::cuda::DeviceArray<DebugData> debug_array;
    if (pars.debug) {
        debug_array.resize(pars.steps);
        debug_array.zero_device();
    }
    cudaDeviceSynchronize(); // ⚠️ testing

    // block/grid size
    dim3 block(pars._block, pars._block);
    dim3 grid((pars._width + block.x - 1) / block.x,
              (pars._height + block.y - 1) / block.y);

    // ================================================================

    for (int step = 0; step < pars.steps; ++step) {

        // if we have rain
        if (pars.rain_rate > 0.0f) {
            rain_pass<<<grid, block, 0, stream.get()>>>(
                dev_pars.dev_ptr(), pars._width, pars._height,

                curand_array_2d.dev_ptr(), // in/out

                rain_map.dev_ptr(), // optional in

                dev_water_map_in // out
            );
        }

        calculate_flux<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(), pars._width, pars._height,
            step,

            dev_height_map_in,   // in
            dev_water_map_in,    // in
            dev_sediment_map_in, // in

            flux8.dev_ptr(),          // out
            sediment_flux8.dev_ptr(), // out
            slope_map.dev_ptr()       // out

        );

        apply_flux<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(), pars._width, pars._height,

            dev_height_map_in,        // in
            dev_water_map_in,         // in
            dev_sediment_map_in,      // in
            flux8.dev_ptr(),          // in
            sediment_flux8.dev_ptr(), // in
            slope_map.dev_ptr(),      // in

            hardness_map.dev_ptr(), // optional in

            dev_height_map_out,  // out
            dev_water_map_out,   // out
            dev_sediment_map_out // out
        );

        if (pars.debug) {
            debug_pass<<<grid, block, 0, stream.get()>>>(
                dev_pars.dev_ptr(), pars._width, pars._height,
                step,

                dev_height_map_out,   // in
                dev_water_map_out,    // in
                dev_sediment_map_out, // in
                debug_array.dev_ptr() // out

            );
        }

        // flip the in/out maps
        std::swap(dev_height_map_in, dev_height_map_out);
        std::swap(dev_water_map_in, dev_water_map_out);
        std::swap(dev_sediment_map_in, dev_sediment_map_out);
    }

    // ================================================================

    stream.sync();           // wait for the stream to finish
    cudaDeviceSynchronize(); // ⚠️ testing

    timer.mark_time();
    printf("calculation time: %.3f seconds\n", timer.elapsed_seconds());

    height_map.download();
    water_map.download();
    sediment_map.download();

    timer.mark_time();
    printf("download time: %.3f seconds\n", timer.elapsed_seconds());

    // report the debug data
    if (pars.debug) {

        float map_size = height_map.size();

        std::vector<DebugData> host_debug(pars.steps);
        cudaMemcpy(host_debug.data(),
                   debug_array.dev_ptr(),
                   pars.steps * sizeof(DebugData),
                   cudaMemcpyDeviceToHost);
        printf("================================================================\n");
        printf("%s\t%s\t%s\t%s\n", "step", "height\n", "water", "sediment");
        printf("----------------------------------------------------------------\n");
        for (int step = 0; step < pars.steps; ++step) {

            if (step % pars.debug_mod != 0)
                continue;

            auto &data = host_debug[step];
            printf("%d\t%.6f\t%.6f\t%.6f\n",
                   step + 1,
                   data.total_height / map_size,
                   data.total_water / map_size,
                   data.total_sediment / map_size);
        }
        printf("================================================================\n");
    }

    deallocate_device();
}

TEMPLATE_CLASS_NAME::TEMPLATE_CLASS_NAME() {
}

TEMPLATE_CLASS_NAME::~TEMPLATE_CLASS_NAME() {
    deallocate_device();
}

#pragma endregion

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
