#include "erosion3.cuh"
#include "noise_util.cuh"
#include <chrono>

#define OFFSET_ORDER 1 // 0 is the orginal WEIRD DOESSNT WORK, MIGHT BE THE OPPOSITE CODE, 1 modified
#define HASH_INT_ORDER

namespace TEMPLATE_NAMESPACE {


    // Apply inverse-square crater imprint onto a heightmap.
// h: heightmap (row-major), W,H: dimensions
// cx,cy: impact center in pixels (float for subpixel)
// k: excavation scale (meters per unit energy)
// r0: softening radius in pixels (prevents singularities)
// mask_radius: optional clamp for finite blast radius
__global__ void crater_imprint(float* h, int W, int H,
                               float cx, float cy,
                               float k, float r0, float mask_radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float dx = (x + 0.5f) - cx;
    float dy = (y + 0.5f) - cy;
    float r2 = dx*dx + dy*dy;

    if (mask_radius > 0.0f && r2 > mask_radius*mask_radius) return;

    float denom = r2 + r0*r0;        // softening
    float E = 1.0f / denom;          // inverse-square
    float dh = -k * E;               // excavation depth

    // Optional: taper center to avoid a pixel spike when r0 is small
    // dh *= (r2 / (r2 + r0*r0));

    int idx = y * W + x;
    h[idx] += dh;
}




__device__ __forceinline__ int wrap_or_clamp(int i, int n, bool wrap) {
    if (wrap) {
        int m = i % n;
        return m < 0 ? m + n : m;
    }
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// calculates the changes that need to occur
__global__ void flux_pass(
    const Parameters pars,
    int width, int height,
    const float *__restrict__ height_in,
    const float *__restrict__ water_in,
    const float *__restrict__ sediment_in,
    // outputs
    float *__restrict__ flux8, // 8 fluxes per cell (neighbor order)
    float *__restrict__ dh_out,
    float *__restrict__ ds_out,
    float *__restrict__ dw_out) {

    //
    //

#if OFFSET_ORDER == 0
    const float d_dist = 1.41421356;
    // const float d_dist = 1.0;
    const int2 offs[8] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    // const float dist[8] = {1, 1, 1, 1, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};
    const float dist[8] = {1, 1, 1, 1, d_dist, d_dist, d_dist, d_dist};
#elif OFFSET_ORDER == 1
    const float d_dist = 1.41421356;
    // const float d_dist = 1.0;
    const int2 offs[8] = {{1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-1, 0}};
    // const float dist[8] = {1, 1, 1, 1, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};
    const float dist[8] = {1, 1, 1, d_dist, d_dist, d_dist, d_dist, 1};
#endif

    //
    //

    //
    //
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    float h = height_in[idx];
    float w = water_in[idx] + pars.rain_rate;
    float s_cur = sediment_in[idx];

    float slopes[8];
    float sum_slope = 0.f;

    // neighbor heights and slopes
    for (int n = 0; n < 8; ++n) {

#ifdef HASH_INT_ORDER
        auto i2 = (n + noise_util::hash_int(x, y, 0)) % 8;
#else
        auto i2 = n;
#endif

        int nx = wrap_or_clamp(x + offs[i2].x, width, pars.wrap);
        int ny = wrap_or_clamp(y + offs[i2].y, height, pars.wrap);
        int nidx = ny * width + nx;
        float nh = height_in[nidx];
        float s = (h - nh) / dist[i2];
        float sd = s > 0.f ? s : 0.f;
        slopes[i2] = sd;
        sum_slope += sd;
    }

    float outflow_cap = fminf(w, pars.w_max);
    float outflow_sum = 0.f;

    // proportional flux
    float *cell_flux = &flux8[idx * 8];
    if (sum_slope > 1e-6f && outflow_cap > 0.f) {
        for (int i = 0; i < 8; ++i) {
            float q = (slopes[i] / sum_slope) * outflow_cap;
            cell_flux[i] = q;
            outflow_sum += q;
        }
    } else {
        for (int i = 0; i < 8; ++i)
            cell_flux[i] = 0.f;
    }

    // velocity proxy and capacity
    float v = 0.f;
    for (int i = 0; i < 8; ++i)
        v += cell_flux[i] * slopes[i];
    float C = pars.capacity * v;

    float erode = 0.f, deposit = 0.f;
    if (C > s_cur) {
        erode = pars.erode * (C - s_cur);
    } else {
        deposit = pars.deposit * (s_cur - C);
    }

    // write deltas (applied in pass B)
    dh_out[idx] = deposit - erode;              // positive = deposition raises height
    ds_out[idx] = erode - deposit;              // sediment increases when eroding, decreases when depositing
    dw_out[idx] = -outflow_sum - pars.evap * w; // water loss: outflow + evaporation
}

// apply the changes
__global__ void apply_pass(
    const Parameters pars,
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

#if OFFSET_ORDER == 0
    const int2 offs[8] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
#elif OFFSET_ORDER == 1
    const int2 offs[8] = {{1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-1, 0}};
#endif

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

#ifndef MODIFY_THIS
#else
    float inflow = 0.f;
#endif

    for (int n = 0; n < 8; ++n) {

#ifdef HASH_INT_ORDER
        auto i2 = (n + noise_util::hash_int(x, y, 0)) % 8;
#else
        auto i2 = n;
#endif

        int nx = x + offs[i2].x;
        int ny = y + offs[i2].y;

        // if (nx < 0 || nx >= width || ny < 0 || ny >= height)
        // continue;

        nx = wrap_or_clamp(x + offs[i2].x, width, pars.wrap); // note we lost continue
        ny = wrap_or_clamp(y + offs[i2].y, height, pars.wrap);

        int nidx = ny * width + nx;
        // opposite direction index (neighbor sending to me)

#ifndef MODIFY_THIS
        int opp = i2 ^ 1; // crude: 0<->1, 2<->3, 4<->6, 5<->7   // TRYING TO FIND OPPOSITE
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

// recomended stuff

// water_in.resize(...);
// water_out.resize(...);
// sediment_in.resize(...);
// sediment_out.resize(...);
// height_in.resize(...);
// height_out.resize(...);

// flux0.resize(width, height, 8); // 8 neighbors per cell
// dh_out.resize(...);
// ds_out.resize(...);
// dw_out.resize(...);

// pars.rain_rate;
// pars.evap_rate;
// pars.w_max;
// pars.k_capacity;
// pars.k_erode;
// pars.k_deposit;
// pars.wrap;
// pars.epsilon;

//
//
//
//

// we are currently talking about erosion here
// https://copilot.microsoft.com/chats/fEfp39jA1SjQeW7yA4D4Y

// but we need more advanced frameworking

void TEMPLATE_CLASS_NAME::process() {

    printf("<<< Erosion Process >>>\n");

    height_map.upload();

    pars.width = height_map.get_width();
    pars.height = height_map.get_height();

    water_map.resize(pars.width, pars.height);
    sediment_map.resize(pars.width, pars.height);
    // flux8.resize(pars.width, pars.height);
    dh_out.resize(pars.width, pars.height);
    ds_out.resize(pars.width, pars.height);
    dw_out.resize(pars.width, pars.height);

    water_map.clear();
    sediment_map.clear();
    // flux8.clear();
    dh_out.clear();
    ds_out.clear();
    dw_out.clear();

    water_map.upload();
    sediment_map.upload();
    // flux8.upload();
    dh_out.upload();
    ds_out.upload();
    dw_out.upload();

    size_t array_size = pars.width * pars.height;

    // allocate a flux map 8x larger than the other maps
    core::CudaArrayManager<float> flux8;
    flux8.resize(array_size * 8);
    flux8.zero_device();

    //
    //

    // out maps, will be freed when we go out of scope
    core::CudaArrayManager<float> height_map_out;
    height_map_out.resize(array_size);
    height_map_out.zero_device();

    core::CudaArrayManager<float> water_map_out;
    water_map_out.resize(array_size);
    water_map_out.zero_device();

    core::CudaArrayManager<float> sediment_map_out;
    sediment_map_out.resize(array_size);
    sediment_map_out.zero_device();

    //
    //
    // core::CudaStruct<Parameters> gpu_pars(pars); // automaticly uploads and free

    // #define X(TYPE, NAME) \
    //     NAME.upload_to_device();
    //     TEMPLATE_CLASS_MAPS
    // #undef X

    core::CudaStruct<Parameters> gpu_pars(pars); // automaticly uploads and free

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);


                  auto start_time = std::chrono::high_resolution_clock::now(); // ⏲️ Timer


    for (int i = 0; i < pars.steps; i += 2) {
        // --- timestep 1: in -> out ---
        flux_pass<<<grid, block>>>(
            pars,
            pars.width, pars.height,
            height_map.dev_ptr(),
            water_map.dev_ptr(),
            sediment_map.dev_ptr(),
            flux8.data(),
            dh_out.dev_ptr(),
            ds_out.dev_ptr(),
            dw_out.dev_ptr());

        apply_pass<<<grid, block>>>(
            pars,
            pars.width, pars.height,
            height_map.dev_ptr(),
            water_map.dev_ptr(),
            sediment_map.dev_ptr(),
            flux8.data(),
            dh_out.dev_ptr(),
            ds_out.dev_ptr(),
            dw_out.dev_ptr(),
            height_map_out.data(),
            water_map_out.data(),
            sediment_map_out.data());

        // --- timestep 2: out -> in ---
        flux_pass<<<grid, block>>>(
            pars,
            pars.width, pars.height,
            height_map_out.data(),
            water_map_out.data(),
            sediment_map_out.data(),
            flux8.data(),
            dh_out.dev_ptr(),
            ds_out.dev_ptr(),
            dw_out.dev_ptr());

        apply_pass<<<grid, block>>>(
            pars,
            pars.width, pars.height,
            height_map_out.data(),
            water_map_out.data(),
            sediment_map_out.data(),
            flux8.data(),
            dh_out.dev_ptr(),
            ds_out.dev_ptr(),
            dw_out.dev_ptr(),
            height_map.dev_ptr(),
            water_map.dev_ptr(),
            sediment_map.dev_ptr());
    }

    auto end_time = std::chrono::high_resolution_clock::now(); // ⏲️ Timer
    std::chrono::duration<double> elapsed = end_time - start_time;
    double seconds = elapsed.count();
    printf("calculation time: %.2f ms\n", seconds * 1000.0); // ⏱️

    height_map.download();
    water_map.download();
    sediment_map.download();

    flux8.free_device();

    // free all maps
#define X(TYPE, NAME)   \
    NAME.free_device(); \
    TEMPLATE_CLASS_MAPS
#undef X
}

} // namespace TEMPLATE_NAMESPACE
