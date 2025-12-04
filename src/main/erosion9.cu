#include "core/cuda/curand_array_2d.cuh"
#include "erosion9.cuh"
#include "erosion9_kernels.cuh"
// #include "noise_util.cuh"
#include "core.h" // timer
#include "cuda_math.cuh"
#include <stdexcept> // std::runtime_error

namespace TEMPLATE_NAMESPACE {

#pragma region HELPERS

// ping pong helper
template <typename T>
__device__ inline T *get_map_ptr(T *in, T *out, int step) {
    return step % 2 == 0 ? in : out;
}

// ping pong helper
template <typename MapPtr>
__device__ inline float read_map_in(MapPtr in, MapPtr out, int step, int idx) {
    return get_map_ptr(in, out, step)[idx];
}
// ping pong helper
template <typename MapPtr>
__device__ inline void write_map_out(MapPtr in, MapPtr out, int step, int idx, float value) {
    get_map_ptr(in, out, step)[idx] = value;
}

__device__ inline int pos_to_idx(int2 pos, int map_width) {
    return pos.y * map_width + pos.x;
}

#pragma endregion

#pragma region KERNELS

// calculate the layer height, set it to height_map and _surface_map
__global__ void calc_layer_height(
    const Parameters *pars,
    const ArrayPtrs *arrays,
    const int step) {
    // // ================================================================
    // int2 map_size = make_int2(pars->_width, pars->_height);
    // int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    // if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
    //     return;
    // int idx = pos_to_idx(pos, map_size.x);
    // // ----------------------------------------------------------------
    // ================================================================
    int map_width = pars->_width;
    int map_height = pars->_height;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height) // bounds check
        return;
    int idx = y * map_width + x;
    // ----------------------------------------------------------------

    auto layer_map = get_map_ptr(arrays->layer_map, arrays->_layer_map_out, step);    // in
    auto height_map = get_map_ptr(arrays->height_map, arrays->_height_map_out, step); // out
    auto water_map = get_map_ptr(arrays->water_map, arrays->_water_map_out, step);    // in
    // auto sediment_map = get_map_ptr(arrays->sediment_map, arrays->_sediment_map_out, step);

    // auto layer_map_out = get_map_ptr(arrays->_layer_map_out, arrays->layer_map, step);
    // auto height_map_out = get_map_ptr(arrays->_height_map_out, arrays->height_map, step);
    // auto water_map_out = get_map_ptr(arrays->_water_map_out, arrays->water_map, step);
    // auto sediment_map_out = get_map_ptr(arrays->_sediment_map_out, arrays->sediment_map, step);

    int layer_count = pars->_layers;
    int layer_idx = idx * layer_count;

    // find height from layers
    float height = 0.0;
    for (int i = 0; i < layer_count; i++) {
        height += layer_map[layer_idx + i];
    }
    float water = water_map[idx];
    float surface = height + water;

    height_map[idx] = height;
    arrays->_surface_map[idx] = surface;
}

// Inputs: difference = Δz to neighbor, h = local water depth,
// scale = cell spacing, is_diag = true for diagonals, n = Manning roughness,
// v_max = cap
__device__ inline float manning_speed(float difference, float h, float scale, bool is_diag, float n, float v_max) {
    // run (horizontal distance)
    float run = is_diag ? (scale * 1.414213562f) : scale;

    // slope (rise/run), downhill only
    float s = difference / run;
    if (s < 0.0f)
        s = 0.0f;

    // hydraulic radius approximation
    float R = h; // tweak if you model roughness/width explicitly
    if (R <= 0.0f || n <= 0.0f)
        return 0.0f;

    // Manning velocity
    float v = (1.0f / n) * powf(R, 2.0f / 3.0f) * sqrtf(s);

    // stability caps
    if (v > v_max)
        v = v_max;
    return v;
}

// new pattern
__global__ void calculate_flux2(
    const Parameters *pars,
    const ArrayPtrs *arrays,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = pos_to_idx(pos, map_size.x);
    // ----------------------------------------------------------------
    int mode = pars->mode;
    // ================================================================
    float height = read_map_in(arrays->height_map, arrays->_height_map_out, step, idx);
    float water = read_map_in(arrays->water_map, arrays->_water_map_out, step, idx);
    float sediment = read_map_in(arrays->sediment_map, arrays->_sediment_map_out, step, idx);

    float surface = height + water;

    // ================================================================
    // [Calculate Slopes]
    // ----------------------------------------------------------------

    float p_slope_gradients[8];
    // float 
    for (int n = 0; n < 8; ++n) {
        int2 new_pos = wrap_or_clamp_index(pos + offsets[n], map_size, pars->wrap);
        int new_idx = pos_to_idx(new_pos, map_size.x);
        float new_height = read_map_in(arrays->height_map, arrays->_height_map_out, step, new_idx);
        float new_water = read_map_in(arrays->water_map, arrays->_height_map_out, step, new_idx);
        float new_sediment = read_map_in(arrays->sediment_map, arrays->_height_map_out, step, new_idx);
        float new_surface = new_height + new_water;

        float difference = surface - new_surface; // positive means we are higher than neighbour

        float horizontal_distance = pars->scale * offset_distances[n];
        float p_slope_gradient = fmaxf(difference / horizontal_distance, 0.0f); // positive slope gradient
        p_slope_gradients[n] = p_slope_gradient;

        // Manning velocity
        float v = (1.0f / n) * powf(water, 2.0f / 3.0f) * sqrtf(p_slope_gradient);

    }
}

/*
Manning’s equation (open channel):

v = (1/n) * R**(2/3) * s**(1/2)
n: roughness (0.02–0.06 typical)
R: hydraulic radius; on grids, approximate with water depth h, or h - bed roughness
Good balance of realism vs cost; still needs clamping.

Fine sediment (mud, silt): Can actually smooth the bed temporarily, lowering
Coarse sediment (gravel, cobbles): Creates rougher boundaries, raising

Smooth concrete channel:
𝑛 ≈ 0.012

Natural streams with sediment and vegetation:
𝑛 ≈ 0.03 – 0.06

Very rough, boulder‑strewn rivers:
𝑛 ≈ 0.07 – 0.15

*/

/*
Shallow-water (Saint-Venant)

https://copilot.microsoft.com/chats/hMzbLGxH7tG1SQkZWEPYW


*/

// // ================================================================
// // won't do this here
// write_map_out(arrays->height_map, arrays->_height_map_out, step, idx, height);
// write_map_out(arrays->water_map, arrays->_water_map_out, step, idx, water);
// write_map_out(arrays->sediment_map, arrays->_sediment_map_out, step, idx, sediment);

#pragma endregion

// 2D Saint-Venant (shallow-water)
#pragma region SHALLOW_WATER

#define HD_INLINE __host__ __device__ inline
constexpr float GRAVITY = 9.81f;

struct Cell {
    float h; // depth
    float u; // vel x
    float v; // vel y
    float z; // bed elevation (terrain)
};

struct Flux3 {
    float fh;  // mass flux
    float fhu; // x-momentum flux
    float fhv; // y-momentum flux
};

HD_INLINE Flux3 flux_x(const Cell &c) {
    float hu = c.h * c.u;
    float hv = c.h * c.v;
    float p = 0.5f * GRAVITY * c.h * c.h; // hydrostatic pressure term
    return {hu, hu * c.u + p, hu * c.v};
}

HD_INLINE Flux3 flux_y(const Cell &c) {
    float hu = c.h * c.u;
    float hv = c.h * c.v;
    float p = 0.5f * GRAVITY * c.h * c.h;
    return {hv, hu * c.v, hv * c.v + p};
}

HD_INLINE float wavespeed(const Cell &c) {
    float c0 = sqrtf(GRAVITY * fmaxf(c.h, 0.0f));
    float ax = fabsf(c.u) + c0;
    float ay = fabsf(c.v) + c0;
    return fmaxf(ax, ay);
}

// Rusanov interface flux and update

HD_INLINE Flux3 rusanov_flux_x(const Cell &L, const Cell &R) {
    Flux3 fL = flux_x(L);
    Flux3 fR = flux_x(R);
    float a = fmaxf(wavespeed(L), wavespeed(R));
    return {
        0.5f * (fL.fh + fR.fh) - 0.5f * a * (R.h - L.h),
        0.5f * (fL.fhu + fR.fhu) - 0.5f * a * ((R.h * R.u) - (L.h * L.u)),
        0.5f * (fL.fhv + fR.fhv) - 0.5f * a * ((R.h * R.v) - (L.h * L.v))};
}

HD_INLINE Flux3 rusanov_flux_y(const Cell &B, const Cell &T) {
    Flux3 gB = flux_y(B);
    Flux3 gT = flux_y(T);
    float a = fmaxf(wavespeed(B), wavespeed(T));
    return {
        0.5f * (gB.fh + gT.fh) - 0.5f * a * (T.h - B.h),
        0.5f * (gB.fhu + gT.fhu) - 0.5f * a * ((T.h * T.u) - (B.h * B.u)),
        0.5f * (gB.fhv + gT.fhv) - 0.5f * a * ((T.h * T.v) - (B.h * B.v))};
}

// Cell update (finite-volume on a uniform grid with spacing dx, dy):

HD_INLINE void update_cell(
    const Flux3 &FxL, const Flux3 &FxR,
    const Flux3 &FyB, const Flux3 &FyT,
    const Cell &c, float n_rough, float dx, float dy, float dt,
    Cell &out) {
    // Divergence of fluxes
    float dh = -(FxR.fh - FxL.fh) / dx - (FyT.fh - FyB.fh) / dy;
    float dhu = -(FxR.fhu - FxL.fhu) / dx - (FyT.fhu - FyB.fhu) / dy;
    float dhv = -(FxR.fhv - FxL.fhv) / dx - (FyT.fhv - FyB.fhv) / dy;

    // Bed slope source terms: -g h ∂z/∂x, -g h ∂z/∂y (use centered diffs outside this helper)
    // Friction (Manning): tau_x, tau_y
    float speed = sqrtf(c.u * c.u + c.v * c.v);
    float h_safe = fmaxf(c.h, 1e-6f);
    float tau_x = GRAVITY * n_rough * n_rough * c.u * speed / powf(h_safe, 4.0f / 3.0f);
    float tau_y = GRAVITY * n_rough * n_rough * c.v * speed / powf(h_safe, 4.0f / 3.0f);

    // Update conserved variables
    float h_new = c.h + dt * dh;
    float hu_new = c.h * c.u + dt * (dhu - h_safe * tau_x);
    float hv_new = c.h * c.v + dt * (dhv - h_safe * tau_y);

    // Reconstruct primitive variables
    h_new = fmaxf(h_new, 0.0f);
    float inv_h = (h_new > 1e-6f) ? (1.0f / h_new) : 0.0f;
    out.h = h_new;
    out.u = hu_new * inv_h;
    out.v = hv_new * inv_h;

    // Note: add bed-slope source terms outside using ∂z/∂x, ∂z/∂y and -g h ∂z
}

// Pseudocode for a 2D kernel launch over interior cells
__global__ void shallow_water_step(const Cell *in, Cell *out,
                                   int W, int H, float dx, float dy,
                                   float dt, float n_rough) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= W - 1 || y >= H - 1)
        return;

    int idx = y * W + x;
    auto C = in[idx];
    auto L = in[idx - 1];
    auto R = in[idx + 1];
    auto B = in[idx - W];
    auto T = in[idx + W];

    // Interface fluxes (left, right, bottom, top)
    Flux3 FxL = rusanov_flux_x(L, C);
    Flux3 FxR = rusanov_flux_x(C, R);
    Flux3 FyB = rusanov_flux_y(B, C);
    Flux3 FyT = rusanov_flux_y(C, T);

    // Update cell
    Cell Cout;
    update_cell(FxL, FxR, FyB, FyT, C, n_rough, dx, dy, dt, Cout);

    // Bed slope sources (optional, outside update_cell)
    // Compute centered gradients of z:
    float dzdx = (R.z - L.z) / (2.0f * dx);
    float dzdy = (T.z - B.z) / (2.0f * dy);
    float hbar = Cout.h;

    // Apply bed slope acceleration: u' += -g * dt * ∂z/∂x, v' similarly
    Cout.u += -GRAVITY * dt * dzdx;
    Cout.v += -GRAVITY * dt * dzdy;

    out[idx] = Cout;
}

#pragma endregion

void TEMPLATE_CLASS_NAME::allocate_device() {
    switch (pars.mode) {
    case 0:
        allocate_device00();
        break;
    case 1:
        allocate_device01();
        break;
    case 2:
        break;
    case 3:
        break;
    }
}

void TEMPLATE_CLASS_NAME::allocate_device00() {
    if (_device_allocated)
        return;

    pars._width = height_map.dimensions()[0];
    pars._height = height_map.dimensions()[1];
    size_t array_size = pars._width * pars._height;

// allocate and zero arrays
#define ALLOCATE_ARRAYS \
    X(water_map)        \
    X(sediment_map)
#define X(NAME)                                   \
    if (NAME.size() != array_size) {              \
        NAME.resize({pars._width, pars._height}); \
        NAME.zero_device();                       \
    }
    ALLOCATE_ARRAYS
#undef X
#undef ALLOCATE_ARRAYS

// allocate arrays
#define ALLOCATE_ARRAYS  \
    X(_height_map_out)   \
    X(_water_map_out)    \
    X(_sediment_map_out) \
    X(_slope_map)
#define X(NAME)                                   \
    if (NAME.size() != array_size) {              \
        NAME.resize({pars._width, pars._height}); \
    }
    ALLOCATE_ARRAYS
#undef X
#undef ALLOCATE_ARRAYS

    // flux output
    _flux8.resize({array_size * 8});
    _sediment_flux8.resize({array_size * 8});

    _device_allocated = true;
}

void TEMPLATE_CLASS_NAME::allocate_device01() {

    if (_device_allocated)
        return;

    bool layers_mode = false;

    if (!layer_map.empty()) {
        layers_mode = true;
        pars._width = layer_map.dimensions()[0];
        pars._height = layer_map.dimensions()[1];
        pars._layers = layer_map.dimensions()[2];
    } else if (!height_map.empty()) {
        pars._width = height_map.dimensions()[0];
        pars._height = height_map.dimensions()[1];
        pars._layers = 0;
    } else {
        throw std::runtime_error("layer_map and height_map empty!");
    }

    size_t array_size = pars._width * pars._height;

    // flux output
    _flux8.resize({array_size * 8});
    _sediment_flux8.resize({array_size * 8});
    _layer_map_out.resize({pars._width, pars._height, pars._layers});

    // allocate and zero arrays
#define ZERO_ARRAYS \
    X(water_map)    \
    X(sediment_map)
#define X(NAME)                                   \
    if (NAME.empty()) {                           \
        NAME.resize({pars._width, pars._height}); \
        NAME.zero_device();                       \
    }
    ZERO_ARRAYS
#undef X
#undef ZERO_ARRAYS

// allocate all remaining arrays
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION)         \
    if (NAME.empty()) {                                \
        NAME.resize_helper(pars._width, pars._height); \
    }
    TEMPLATE_CLASS_DEVICE_ARRAY_NS
#undef X

    _device_allocated = true;
}

void TEMPLATE_CLASS_NAME::process() {

    switch (pars.mode) {
    case 0:
        process00();
        break;
    case 1:
        process01();
        break;
    case 2:
        break;
    case 3:
        break;
    }
}

void TEMPLATE_CLASS_NAME::process00() {

    allocate_device();
    configure_device();
    stream.sync();

    core::util::Timer timer;

    for (int step = 0; step < pars.steps; ++step) {

        // if we have rain
        if (pars.rain_rate > 0.0f) {
            rain_pass<<<grid, block, 0, stream.get()>>>(
                dev_pars.dev_ptr(), pars._width, pars._height,

                nullptr,

                rain_map.dev_ptr(), // optional in

                water_map.dev_ptr() // out
            );
        }

        calculate_flux<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(), pars._width, pars._height,
            step,

            height_map.dev_ptr(),   // in
            water_map.dev_ptr(),    // in
            sediment_map.dev_ptr(), // in

            _flux8.dev_ptr(),          // out
            _sediment_flux8.dev_ptr(), // out
            _slope_map.dev_ptr()       // out

        );

        apply_flux<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(), pars._width, pars._height,

            height_map.dev_ptr(),      // in
            water_map.dev_ptr(),       // in
            sediment_map.dev_ptr(),    // in
            _flux8.dev_ptr(),          // in
            _sediment_flux8.dev_ptr(), // in
            _slope_map.dev_ptr(),      // in

            _height_map_out.dev_ptr(),  // out
            _water_map_out.dev_ptr(),   // out
            _sediment_map_out.dev_ptr() // out
        );

        // flip the in/out maps
        std::swap(height_map, _height_map_out);
        std::swap(water_map, _water_map_out);
        std::swap(sediment_map, _sediment_map_out);
    }

    timer.mark_time();
    pars._calculation_time = timer.elapsed_seconds();
    if (pars.debug_print) {
        printf("⏱️ calculation time: %.3f seconds\n", timer.elapsed_seconds());
        // core::logging::printf("⏱️ calculation time: %.3f seconds\n", timer.elapsed_seconds());
    }
}

void TEMPLATE_CLASS_NAME::process01() {

    allocate_device();
    configure_device();
    stream.sync();

    core::cuda::DeviceStruct<ArrayPtrs> dev_array_ptrs(get_array_ptrs()); // device side pars

    for (int step = 0; step < pars.steps; ++step) {
        // calculate_flux2<<<grid, block, 0, stream.get()>>>(dev_pars.dev_ptr(), dev_array_ptrs.dev_ptr(), step);
    }
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
