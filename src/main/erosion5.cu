/*
Notes

Hydraulic erosion: water carries sediment downhill, depositing when flow slows

Aeolian erosion: wind lifts particles, moves them across terrain, and deposits them when wind speed decreases or slope traps them.

Slope stability Sand tends to accumulate until the slope exceeds the angle of repose (~30–35° for dry sand).
If exceeded, grains slide downhill until equilibrium is restored.


*/

#include "core.h"
#include "erosion5.cuh"
#include "noise_util.cuh"
#include <chrono>
#include <cmath>

namespace TEMPLATE_NAMESPACE {

constexpr float SQRT2 = 1.4142135623730950488f;

// 0=E, 1=W, 2=N, 3=S, 4=NE, 5=NW, 6=SE, 7=SW
__device__ __constant__ int2 offsets[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
__device__ __constant__ float offset_distances[8] = {1.0f, 1.0f, 1.0f, 1.0f, SQRT2, SQRT2, SQRT2, SQRT2};
__device__ __constant__ int opposite_offsets[8] = {1, 0, 3, 2, 5, 4, 7, 6};

__device__ __constant__ int2 offsets4[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

__device__ __forceinline__ int wrap_or_clamp(int i, int n, bool wrap) {
    if (wrap) {
        int m = i % n;
        return m < 0 ? m + n : m;
    } else {
        return i < 0 ? 0 : (i >= n ? n - 1 : i);
    }
}

// positive modulo wrap (note it might be faster to concider other wrap methods)
__device__ __forceinline__ int posmod(int value, int mod) {
    int result = value % mod;
    return result < 0 ? result + mod : result;
}

__global__ void rain_pass(
    const Parameters *pars,
    const int map_width, const int map_height,
    curandState *rand_states,
    float *__restrict__ water_map

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

// 8-neighbor flow calculation (no erosion, no sediment)
__global__ void calculate_flux(
    const Parameters *pars,
    const int map_width, const int map_height,
    const int step,

    const float *__restrict__ height_map,
    const float *__restrict__ water_map,

    float *__restrict__ flux8 // 8 fluxes per cell
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
    float surface = height + water;

    float slopes[8];        // positive slope to neighbours
    float sum_slope = 0.0f; // total slope

    // 1. calculate slopes
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

        // ❓ make sure diagonals are further, normally true
        if (pars->correct_diagonal_distance) {
            difference /= offset_distances[n];
        }

        float sd = difference > 0.0f ? difference : 0.0f; // amount higher we are than neighbour (or 0 if we are lower)

        //

        slopes[n] = sd;
        sum_slope += sd;
    }

    // 2. slope based flux
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

    // 3. Add isotropic diffusion, recommend ~0.001
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
}

/*

Each cell stores 8 outflows, one per neighbor direction.

When you apply flow, you subtract your own outflows.

To add inflows, you look at each neighbor and ask: which of its flux slots points toward me?

That’s the “opposite” index. Without it, you’d never add incoming water, so everything just drains away.

*/

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return min(max(v, lo), hi);
}

// WARNING USED AI FOR THIS CONFUSING BIT
__global__ void apply_flux(
    const Parameters *pars,
    const int map_width, const int map_height,

    const float *__restrict__ height_map,
    const float *__restrict__ water_map,
    const float *__restrict__ sediment_map,
    const float *__restrict__ flux8, // 8 fluxes per cell

    float *__restrict__ height_map_out,
    float *__restrict__ water_map_out,
    float *__restrict__ sediment_map_out

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

    // subtract my outflows
    float outflow_sum = 0.f;
    for (int n = 0; n < 8; ++n) {
        outflow_sum += flux8[idx * 8 + n];
    }
    water -= outflow_sum;

    // optional carving by outflow
    if (pars->outflow_erode > 0.0f) {
        height -= pars->outflow_erode * outflow_sum;
    }

    // add inflows from neighbors
    float inflow_sum = 0.f;
    for (int n = 0; n < 8; ++n) {
        int nx = wrap_or_clamp(x + offsets[n].x, map_width, pars->wrap);
        int ny = wrap_or_clamp(y + offsets[n].y, map_height, pars->wrap);
        int nidx = ny * map_width + nx;

        int opp = opposite_offsets[n];
        inflow_sum += flux8[nidx * 8 + opp];
    }
    water += inflow_sum;

    water -= pars->evaporation_rate; // evaporation

    // optional carving by inflow
    if (pars->inflow_erode > 0.0f) {
        height -= pars->inflow_erode * inflow_sum;
    }

    height = clampf(height, pars->min_height, pars->max_height); // clamp height

    height_map_out[idx] = height;
    water_map_out[idx] = fmaxf(0.f, water);       // no negative water
    sediment_map_out[idx] = fmaxf(0.f, sediment); // no negative sediment
}

//
//
//
//

void TEMPLATE_CLASS_NAME::allocate_device() {

    if (device_allocated)
        return;

    device_allocated = true;

    height_map.upload();
    pars.width = height_map.get_width();
    pars.height = height_map.get_height();

    if (water_map.size() != height_map.size()) { // if we don't start with a map clear
        water_map.resize(pars.width, pars.height);
        water_map.clear();
    }
    water_map.upload();

    if (sediment_map.size() != height_map.size()) { // if we don't start with a map clear
        sediment_map.resize(pars.width, pars.height);
        sediment_map.clear();
    }
    sediment_map.upload();

    // ================================================================

    size_t array_size = pars.width * pars.height;

    // private device arrays
#define X(TYPE, NAME, Z_SIZE)        \
    NAME.resize(array_size *Z_SIZE); \
    NAME.zero_device();
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X

    //     cudaDeviceSynchronize();
}

void TEMPLATE_CLASS_NAME::deallocate_device() {

    auto timer = core::Timer();

    device_allocated = false;

    // free maps
#define X(TYPE, NAME)   \
    NAME.free_device(); \
    TEMPLATE_CLASS_MAPS
#undef X

    // free device arrays
#define X(TYPE, NAME, Z_SIZE) \
    NAME.free_device();
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X

    timer.mark_time();
    printf("deallocate device time: %.3f seconds\n", timer.elapsed_seconds());
}

void TEMPLATE_CLASS_NAME::process() {

    printf("<<< Erosion Process >>>\n");

    auto timer = core::Timer();

    allocate_device();
    core::cuda::Stream stream;
    core::cuda::Struct<Parameters> dev_pars(pars); // automaticly uploads and free

    auto curand_array_2d = core::cuda::CurandArray2D(pars.width, pars.height, stream.get());
    stream.sync(); // important??

    timer.mark_time();
    printf("allocate device time: %.3f seconds\n", timer.elapsed_seconds());

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    // pointers for swapping the maps around (ping/pong)
    float *dev_height_map_in = height_map.dev_ptr();
    float *dev_water_map_in = water_map.dev_ptr();
    float *dev_sediment_map_in = sediment_map.dev_ptr();

    float *dev_height_map_out = height_map_out.dev_ptr();
    float *dev_water_map_out = water_map_out.dev_ptr();
    float *dev_sediment_map_out = sediment_map_out.dev_ptr();

    cudaDeviceSynchronize(); // allocating memory might need a sync?

    // ================================================================

    for (int step = 0; step < pars.steps; ++step) {

        // if we have rain
        if (pars.rain_rate > 0.0f) {
            rain_pass<<<grid, block, 0, stream.get()>>>(
                dev_pars.dev_ptr(), pars.width, pars.height,
                curand_array_2d.dev_ptr(),
                dev_water_map_in);
        }

        calculate_flux<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(), pars.width, pars.height,
            step,

            dev_height_map_in,
            dev_water_map_in,
            flux8.dev_ptr());

        apply_flux<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(), pars.width, pars.height,

            dev_height_map_in,
            dev_water_map_in,
            dev_sediment_map_in,
            flux8.dev_ptr(),

            dev_height_map_out,
            dev_water_map_out,
            dev_sediment_map_out);

        std::swap(dev_height_map_in, dev_height_map_out);
        std::swap(dev_water_map_in, dev_water_map_out);
        std::swap(dev_sediment_map_in, dev_sediment_map_out);
    }

    // ================================================================

    stream.sync(); // wait for the stream to finish

    timer.mark_time();
    printf("calculation time: %.3f seconds\n", timer.elapsed_seconds());

    height_map.download();
    water_map.download();
    sediment_map.download();

    timer.mark_time();
    printf("download time: %.3f seconds\n", timer.elapsed_seconds());

    // deallocate_device();
}

TEMPLATE_CLASS_NAME::TEMPLATE_CLASS_NAME() {
}

TEMPLATE_CLASS_NAME::~TEMPLATE_CLASS_NAME() {
    deallocate_device();
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macros_undef.h"
