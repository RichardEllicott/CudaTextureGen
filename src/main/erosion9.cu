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
    // ================================================================
    // [Rain]
    // ----------------------------------------------------------------
    float rain = pars->rain_rate;
    // if (arrays->rain_map) {
    //     rain *= arrays->rain_map[idx]; // multiply by rain_map if != nullptr
    // }
    arrays->water_map[idx] += rain;
    // ================================================================
    // float height = read_map_in(arrays->height_map, arrays->_height_map_out, step, idx);
    // float water = read_map_in(arrays->water_map, arrays->_water_map_out, step, idx);
    // float sediment = read_map_in(arrays->sediment_map, arrays->_sediment_map_out, step, idx);
    float height = arrays->height_map[idx];
    float water = arrays->water_map[idx];
    float sediment = arrays->sediment_map[idx];
    // ================================================================
    // [Calculate Flows]
    // ----------------------------------------------------------------
    float surface = height + water;
    float outflows[8];
    float total_outflow = 0.0f;

    for (int n = 0; n < 8; ++n) {
        int2 new_pos = wrap_or_clamp_index(pos + offsets[n], map_size, pars->wrap);
        int new_idx = pos_to_idx(new_pos, map_size.x);
        float new_height = read_map_in(arrays->height_map, arrays->_height_map_out, step, new_idx);
        float new_water = read_map_in(arrays->water_map, arrays->_height_map_out, step, new_idx);
        float new_sediment = read_map_in(arrays->sediment_map, arrays->_height_map_out, step, new_idx);
        float new_surface = new_height + new_water;

        float slope_height = surface - new_surface;                                      // positive means we are higher than neighbour
        float horizontal_distance = pars->scale * offset_distances[n];                   // distance to tile (1.0 | ~1.41) * map scale
        float positive_slope_gradient = fmaxf(slope_height / horizontal_distance, 0.0f); // positive slope gradient

        // ----------------------------------------------------------------
        // Manning velocity approx
        // float v = (1.0f / roughness) * powf(water, 2.0f / 3.0f) * sqrtf(p_slope_gradient);
        float outflow = powf(water, 2.0f / 3.0f) * sqrtf(positive_slope_gradient); // based on manning with no roughness
        outflow *= pars->flow_rate;                                                // scale with a flow rate, lowering this number will slow all flow
        outflows[n] = outflow;
        total_outflow += outflow;
    }

    // ================================================================
    // [Calculate Flux]
    // ----------------------------------------------------------------

    // get a scale factor in the case we have exceeded the max water outflow
    // ❓ we could approach this problem different and instead use a max outflow per cell
    float flux_scale = 1.0f;
    if (total_outflow > pars->max_water_outflow) {
        flux_scale = pars->max_water_outflow / total_outflow;
    }

    // pointer's to the flux arrays
    float *_flux8_ptr = &arrays->_flux8[idx * 8];
    float *_sediment_flux8_ptr = &arrays->_sediment_flux8[idx * 8];             // pointer to sediment flux
    float sediment_concentration = (water > 1e-6f) ? (sediment / water) : 0.0f; // sediment concentration
    sediment_concentration *= pars->sediment_capacity;                          // the amount of sediment to transport based on capacity

    for (int n = 0; n < 8; ++n) {
        float scaled_outflow = outflows[n] * flux_scale;
        _flux8_ptr[n] = scaled_outflow;                                   // final outflow value scaled
        _sediment_flux8_ptr[n] = scaled_outflow * sediment_concentration; // final sediment flow
    }
}

__global__ void apply_flux2(
    const Parameters *pars,
    const ArrayPtrs *arrays,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = pos_to_idx(pos, map_size.x);
    // ================================================================
    // float height = read_map_in(arrays->height_map, arrays->_height_map_out, step, idx);
    // float water = read_map_in(arrays->water_map, arrays->_water_map_out, step, idx);
    // float sediment = read_map_in(arrays->sediment_map, arrays->_sediment_map_out, step, idx);
    float height = arrays->height_map[idx];
    float water = arrays->water_map[idx];
    float sediment = arrays->sediment_map[idx];
    // ================================================================
    // [Calculate Flows]
    // ----------------------------------------------------------------

    float water_inflow = 0.f;     // inflow calculated by visting neighbours
    float sediment_change = 0.0f; // same with sediment change

    float water_outflow = 0.0f; // ❓ could be more effecient to calculate this previously (like as a delta)

    for (int n = 0; n < 8; ++n) {
        water_outflow += arrays->_flux8[idx * 8 + n]; // ❓ could be more effecient to calculate this previously (like as a delta)

        int2 new_pos = wrap_or_clamp_index(pos + offsets[n], map_size, pars->wrap);
        int new_idx = pos_to_idx(new_pos, map_size.x);
        int opposite_offset = opposite_offset_refs[n];

        water_inflow += arrays->_flux8[new_idx * 8 + opposite_offset]; // ❓  confusing

        sediment_change -= arrays->_sediment_flux8[idx * 8 + n];               // outflow
        sediment_change += arrays->_sediment_flux8[new_idx * 8 + opposite_offset]; // inflow
    }

    water -= water_outflow;
    water += water_inflow;
    sediment += sediment_change;

    float erosion = water_outflow * pars->erosion_rate;
    float available_erosion = height - pars->min_height; // limit erosion to available rock above min_height
    erosion = fminf(erosion, fmaxf(0.0f, available_erosion));

    sediment += erosion; // ❓ scale this by the material, some might make less sediment
    height -= erosion;

    // height = clamp(height, pars->min_height, pars->max_height); //❓❓ pointless won't go up?

    water -= pars->evaporation_rate; // evaporation

    // ================================================================
    // [sediment_change]  ❓ can intergrate with previous
    // ----------------------------------------------------------------

    // for (int n = 0; n < 8; ++n) {
    //     int2 new_pos = wrap_or_clamp_index(pos + offsets[n], map_size, pars->wrap);
    //     int new_idx = pos_to_idx(new_pos, map_size.x);
    //     int opposite_offset = opposite_offset_refs[n];

    // }

    // ================================================================
    // [Deposition]
    // ----------------------------------------------------------------
    float deposit = sediment * pars->deposition_rate; // ❓ simple, no capacity
    sediment -= deposit;
    height += deposit;
    // ================================================================
    // [Drain]
    // ----------------------------------------------------------------
    if (pars->drain_at_min_height && height <= pars->min_height) {
        water = 0.0f;
        sediment = 0.0f;
    }
    // ================================================================
    // [output]
    // ----------------------------------------------------------------

    arrays->height_map[idx] = height;
    arrays->water_map[idx] = fmaxf(0.f, water);       // no negative water
    arrays->sediment_map[idx] = fmaxf(0.f, sediment); // no negative sediment
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
    // _layer_map_out.resize({pars._width, pars._height, pars._layers});

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

    // ================================================================
    // allocate arrays
    // #define ALLOCATE_ARRAYS  \
//     X(_height_map_out)   \
//     X(_water_map_out)    \
//     X(_sediment_map_out) \
//     X(_slope_map)
    // #define X(NAME)                                   \
//     if (NAME.size() != array_size) {              \
//         NAME.resize({pars._width, pars._height}); \
//     }
    //     ALLOCATE_ARRAYS
    // #undef X
    // #undef ALLOCATE_ARRAYS

// ================================================================
// ⚠️ allocate all remaining arrays  ... WARNING I SEEM TO NEED THIS??
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

    // printf("process01...");

    allocate_device();
    configure_device();
    stream.sync();

    core::cuda::DeviceStruct<ArrayPtrs> dev_array_ptrs(get_array_ptrs()); // device side pars

    for (int step = 0; step < pars.steps; ++step) {
        calculate_flux2<<<grid, block, 0, stream.get()>>>(dev_pars.dev_ptr(), dev_array_ptrs.dev_ptr(), step);
        apply_flux2<<<grid, block, 0, stream.get()>>>(dev_pars.dev_ptr(), dev_array_ptrs.dev_ptr(), step);
    }

    stream.sync();
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
