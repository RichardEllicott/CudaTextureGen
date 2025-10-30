#include "erosion3.cuh"
#include "noise_util.cuh"
#include "core.h"
#include <chrono>
#include <cmath>

// 0 is the orginal (broken!)
// 1 works, but does't even seem to be opposite
// 2 doesn't seem to give a good result but should be technically correct
#define EROSION3_OFFSET_ORDER_HACK 1

namespace TEMPLATE_NAMESPACE {


// constexpr float SQRT2 = 1.41421356f;
constexpr float SQRT2 = 1.4142135623730950488f; // square root of 2 (diagonal accross a square)
constexpr float DIAG_DIST = SQRT2;

#if EROSION3_OFFSET_ORDER_HACK == 0
__device__ __constant__ int2 offsets[8] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
__device__ __constant__ float offset_distances[8] = {1, 1, 1, 1, DIAG_DIST, DIAG_DIST, DIAG_DIST, DIAG_DIST};
#elif EROSION3_OFFSET_ORDER_HACK == 1
__device__ __constant__ int2 offsets[8] = {{1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-1, 0}};
__device__ __constant__ float offset_distances[8] = {1.0f, 1.0f, 1.0f, DIAG_DIST, DIAG_DIST, DIAG_DIST, DIAG_DIST, 1.0f};
#elif EROSION3_OFFSET_ORDER_HACK == 2
__device__ __constant__ int2 offsets[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
__device__ __constant__ float offset_distances[8] = {1, 1, 1, 1, DIAG_DIST, DIAG_DIST, DIAG_DIST, DIAG_DIST};
#endif

__device__ __forceinline__ int wrap_or_clamp(int i, int n, bool wrap) {
    if (wrap) {
        int m = i % n;
        return m < 0 ? m + n : m;
    }
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// positive modulo wrap (note it might be faster to concider other wrap methods)
__device__ __forceinline__ int posmod(int value, int mod) {
    int result = value % mod;
    return result < 0 ? result + mod : result;
}

// calculates the changes that need to occur
__global__ void flux_pass(
    const Parameters *pars,
    int map_width, int map_height,
    const float *__restrict__ height_in,
    const float *__restrict__ water_in,
    const float *__restrict__ sediment_in,
    // outputs
    float *__restrict__ flux8, // 8 fluxes per cell (neighbor order)
    float *__restrict__ dh_out,
    float *__restrict__ ds_out,
    float *__restrict__ dw_out) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height)
        return;

    int idx = y * map_width + x;
    float cell_height = height_in[idx];
    float water = water_in[idx] + pars->rain_rate;
    float sediment = sediment_in[idx];

    float slopes[8];
    float sum_slope = 0.f;

    // neighbor heights and slopes
    for (int n = 0; n < 8; ++n) {

        int i = pars->debug_hash_cell_order
                    ? (n + noise_util::hash_int(x, y, 0)) % 8
                    : n;

        int nx = wrap_or_clamp(x + offsets[i].x, map_width, pars->wrap);
        int ny = wrap_or_clamp(y + offsets[i].y, map_height, pars->wrap);
        int nidx = ny * map_width + nx;
        float nh = height_in[nidx];
        float s = (cell_height - nh) / offset_distances[i];
        float sd = s > 0.f ? s : 0.f;
        slopes[i] = sd;
        sum_slope += sd;
    }

    float max_outflow_this_step = fminf(water, pars->max_water_outflow);
    float outflow_sum = 0.f;

    const float EPSILON_SLOPE = 1e-6f; // prevents a divide by zero issue

    // proportional flux
    float *cell_flux = &flux8[idx * 8];
    if (sum_slope > EPSILON_SLOPE && max_outflow_this_step > 0.f) {
        for (int i = 0; i < 8; ++i) {
            float q = (slopes[i] / sum_slope) * max_outflow_this_step;
            cell_flux[i] = q;
            outflow_sum += q;
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            cell_flux[i] = 0.f; // no flux at all
        }
    }

    // velocity proxy and capacity
    float velocity_proxy = 0.f;
    for (int i = 0; i < 8; ++i)
        velocity_proxy += cell_flux[i] * slopes[i];
    float sediment_capacity = pars->capacity * velocity_proxy; // C is the sediment carrying capacity of the water in this cell. (faster velocity is more sediment)

    float erode = 0.f, deposit = 0.f;
    if (sediment_capacity > sediment) {
        erode = pars->erode * (sediment_capacity - sediment);
    } else {
        deposit = pars->deposit * (sediment - sediment_capacity);
    }

    // write deltas (applied in pass B)
    dh_out[idx] = deposit - erode;                               // positive = deposition raises height
    ds_out[idx] = erode - deposit;                               // sediment increases when eroding, decreases when depositing
    dw_out[idx] = -outflow_sum - pars->evaporation_rate * water; // water loss: outflow + evaporation
}

// apply the changes
__global__ void apply_pass(
    const Parameters *pars,
    int width, int height,
    const float *__restrict__ water_in,
    const float *__restrict__ sediment_in,
    const float *__restrict__ height_in,
    const float *__restrict__ flux, // 8 fluxes per cell
    const float *__restrict__ dh,   // erosion/deposition delta
    const float *__restrict__ ds,   // sediment delta
    const float *__restrict__ dw,   // water delta (loss)
    float *__restrict__ water_out,
    float *__restrict__ sediment_out,
    float *__restrict__ height_out) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    float w = water_in[idx];
    float s = sediment_in[idx];
    float h = height_in[idx];

    // apply local deltas
    w += dw[idx];
    s += ds[idx];
    h += dh[idx];

    // add incoming flux from neighbors
    float inflow = 0.f;

    // #define MODIFY_THIS

    // #ifndef MODIFY_THIS
    // #else
    //     float inflow = 0.f;
    // #endif

    for (int n = 0; n < 8; ++n) {

        int i = pars->debug_hash_cell_order
                    ? (n + noise_util::hash_int(x, y, 0)) % 8
                    : n;

        int nx = x + offsets[i].x;
        int ny = y + offsets[i].y;

        // if (nx < 0 || nx >= width || ny < 0 || ny >= height)
        // continue;

        nx = wrap_or_clamp(x + offsets[i].x, width, pars->wrap); // note we lost continue
        ny = wrap_or_clamp(y + offsets[i].y, height, pars->wrap);

        int nidx = ny * width + nx;
        // opposite direction index (neighbor sending to me)

#ifndef MODIFY_THIS
        int opp = i ^ 1; // crude: 0<->1, 2<->3, 4<->6, 5<->7   // TRYING TO FIND OPPOSITE
        inflow += flux[nidx * 8 + opp];
#else
        int opp_i = opp[n];
        float q_in = flux[nidx * 8 + opp_i];
        float conc = sediment_in[nidx] / fmaxf(water_in[nidx], 1e-6f);
        inflow_s += q_in * conc;

#endif
    }
    w += inflow;

    // write out
    water_out[idx] = fmaxf(0.f, w);
    sediment_out[idx] = fmaxf(0.f, s);
    height_out[idx] = fmaxf(0.f, h);
}

void TEMPLATE_CLASS_NAME::allocate_device() {

    if (device_allocated)
        return;

    device_allocated = true;

    height_map.upload();

    pars.width = height_map.get_width();
    pars.height = height_map.get_height();

    water_map.resize(pars.width, pars.height);
    sediment_map.resize(pars.width, pars.height);
    dh_out.resize(pars.width, pars.height);
    ds_out.resize(pars.width, pars.height);
    dw_out.resize(pars.width, pars.height);

    water_map.clear();
    sediment_map.clear();
    dh_out.clear();
    ds_out.clear();
    dw_out.clear();

    water_map.upload();
    sediment_map.upload();
    dh_out.upload();
    ds_out.upload();
    dw_out.upload();

    size_t array_size = pars.width * pars.height;

    // resize private arrays
    flux8.resize(array_size * 8);
    height_map_out.resize(array_size);
    water_map_out.resize(array_size);
    sediment_map_out.resize(array_size);

    height_map_out.zero_device();
    water_map_out.zero_device();
    sediment_map_out.zero_device();
    flux8.zero_device();

    h_cur = height_map.dev_ptr();
    w_cur = water_map.dev_ptr();
    s_cur = sediment_map.dev_ptr();

    h_next = height_map_out.dev_ptr();
    w_next = water_map_out.dev_ptr();
    s_next = sediment_map_out.dev_ptr();
}

void TEMPLATE_CLASS_NAME::deallocate_device() {

    auto timer = core::Timer();

    device_allocated = false;

    // free all macroed maps
#define X(TYPE, NAME)   \
    NAME.free_device(); \
    TEMPLATE_CLASS_MAPS
#undef X

    // free the extra arrays
    height_map_out.free_device();
    water_map_out.free_device();
    sediment_map_out.free_device();
    flux8.free_device();

    h_cur = nullptr;
    w_cur = nullptr;
    s_cur = nullptr;

    h_next = nullptr;
    w_next = nullptr;
    s_next = nullptr;

    timer.mark_time();
    printf("deallocate device time: %.3f seconds\n", timer.elapsed_seconds());
}

void TEMPLATE_CLASS_NAME::process() {

    printf("<<< Erosion Process >>>\n");

    auto timer = core::Timer();

    allocate_device();
    core::CudaStream stream;
    core::CudaStruct<Parameters> gpu_pars(pars); // automaticly uploads and free

    timer.mark_time();
    printf("allocate device time: %.3f seconds\n", timer.elapsed_seconds());

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    for (int i = 0; i < pars.steps; ++i) {
        // calculate changes
        flux_pass<<<grid, block, 0, stream.get()>>>(
            gpu_pars.dev_ptr(),
            pars.width, pars.height,
            h_cur, w_cur, s_cur,
            flux8.dev_ptr(),
            dh_out.dev_ptr(), ds_out.dev_ptr(), dw_out.dev_ptr());

        // apply changes
        apply_pass<<<grid, block, 0, stream.get()>>>(
            gpu_pars.dev_ptr(),
            pars.width, pars.height,
            h_cur, w_cur, s_cur,
            flux8.dev_ptr(),
            dh_out.dev_ptr(), ds_out.dev_ptr(), dw_out.dev_ptr(),
            h_next, w_next, s_next);

        // swap roles
        std::swap(h_cur, h_next);
        std::swap(w_cur, w_next);
        std::swap(s_cur, s_next);

        _count++;
    }

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
