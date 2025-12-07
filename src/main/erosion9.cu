#include "core/cuda/curand_array_2d.cuh"
#include "erosion9.cuh"
#include "erosion9_kernels.cuh"
// #include "noise_util.cuh"
#include "core.h" // timer
#include "cuda_math.cuh"
#include <stdexcept> // std::runtime_error

#define EROSION_OUTFLOW_PRECALCULATION

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

#pragma region KERNELS2

// KEPT AS NOTE
// 🚧 calculate the layer height, set it to height_map and _surface_map
__global__ void calc_layer_height(
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

    auto layer_map = arrays->layer_map;   // in
    auto height_map = arrays->height_map; // out
    auto water_map = arrays->water_map;   // in

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

// new pattern
__global__ void calculate_flux2(
    const Parameters *pars,
    const ArrayPtrs *arrays,
    DebugOutputs *debug,
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
    if (arrays->rain_map) {
        rain *= arrays->rain_map[idx]; // multiply by rain_map if != nullptr
    }
    arrays->water_map[idx] += rain;
    if (pars->_debug) {
        atomicAdd(&(debug->_debug_rain_total), rain);
    }
    // ================================================================
    float height = arrays->height_map[idx];
    float water = arrays->water_map[idx];
    float sediment = arrays->sediment_map[idx];
    // ================================================================
    // [Calculate Flows]
    // ----------------------------------------------------------------
    float surface_height = height + water;
    float fluxes[8];
    float total_flux = 0.0f;

    if (pars->slope_jitter > 0.0f) {
        surface_height += hash_float_signed(pos.x, pos.y, step, 1234) * pars->slope_jitter; // ⚠️ we change the height here as this is the cheapest jitter (do not write back)
    }

    for (int n = 0; n < 8; ++n) {
        int2 new_pos = wrap_or_clamp_index(pos + offsets[n], map_size, pars->wrap);
        int new_idx = pos_to_idx(new_pos, map_size.x);

        float new_height = arrays->height_map[new_idx];
        float new_water = arrays->water_map[new_idx];
        float new_sediment = arrays->sediment_map[new_idx];

        float new_surface_height = new_height + new_water;

        float slope_height = surface_height - new_surface_height;                        // positive means we are higher than neighbour
        float horizontal_distance = pars->scale * offset_distances[n];                   // distance to tile (1.0 | ~1.41) * map scale
        float positive_slope_gradient = fmaxf(slope_height / horizontal_distance, 0.0f); // positive slope gradient

        positive_slope_gradient = min(positive_slope_gradient, pars->positive_slope_gradient_cap); // 🚧 new cap

        // ----------------------------------------------------------------
        // Manning velocity approx
        // float v = (1.0f / roughness) * powf(water, 2.0f / 3.0f) * sqrtf(p_slope_gradient);
        float flux = powf(water, 2.0f / 3.0f) * sqrtf(positive_slope_gradient); // based on manning with no roughness
        flux *= pars->flow_rate;                                                // scale with a flow rate, lowering this number will slow all flow
        fluxes[n] = flux;
        total_flux += flux;
    }

// #define CODE_ROUTE 1
#if CODE_ROUTE == 0

    // ================================================================
    // [Calculate Flux]
    // ----------------------------------------------------------------

    float max_or_total_water = min(water, pars->max_water_outflow); // firstly either the water or max outflow
    float flux_scale = 1.0f;                                        // will scale the resulting flux by this to ensure water total doesn't exceed max_or_total_water
    if (total_flux > max_or_total_water) {
        flux_scale = max_or_total_water / total_flux;
    }

    // pointer's to the flux arrays
    float *_flux8_ptr = &arrays->_flux8[idx * 8];
    float *_sediment_flux8_ptr = &arrays->_sediment_flux8[idx * 8];                                       // pointer to sediment flux
    float sediment_concentration = (max_or_total_water > 1e-6f) ? (sediment / max_or_total_water) : 0.0f; // sediment concentration
    sediment_concentration *= pars->sediment_capacity;                                                    // the amount of sediment to transport based on capacity

    for (int n = 0; n < 8; ++n) {
        float scaled_outflow = fluxes[n] * flux_scale;
        _flux8_ptr[n] = scaled_outflow;                                   // final outflow value scaled
        _sediment_flux8_ptr[n] = scaled_outflow * sediment_concentration; // final sediment flow
    }

#ifdef EROSION_OUTFLOW_PRECALCULATION
    // OPTIONAL OPTIMIZATION (save total outflow from this cell)
    total_flux *= flux_scale;
    arrays->_water_map_out[idx] = total_flux;
    arrays->_sediment_map_out[idx] = total_flux * sediment_concentration;
#endif

#elif CODE_ROUTE == 1

    // ================================================================
    // [Calculate Outflow] BROKEN
    // ----------------------------------------------------------------

    float *_flux8_ptr = &arrays->_flux8[idx * 8];
    float max_or_total_water = min(water, pars->max_water_outflow); // firstly either the water or max outflow
    // normalize so total is max_or_total_water
    for (int n = 0; n < 8; ++n) {
        // water outflow
        float flux = outflows[n];
        flux /= total_outflow;      // total now 1
        flux *= max_or_total_water; // total now = to max_or_total_water
        _flux8_ptr[n] = flux;       // final outflow value scaled
    }

#else
#error "Unsupported CODE_ROUTE value"
#endif
#undef CODE_ROUTE
}

__global__ void apply_flux2(
    const Parameters *pars,
    const ArrayPtrs *arrays,
    DebugOutputs *debug,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = pos_to_idx(pos, map_size.x);
    // ================================================================
    float height = arrays->height_map[idx];
    float water = arrays->water_map[idx];
    float sediment = arrays->sediment_map[idx];
    // ================================================================
    // [Calculate Flows]
    // ----------------------------------------------------------------

#ifdef EROSION_OUTFLOW_PRECALCULATION
    float water_outflow = arrays->_water_map_out[idx]; // load precalculated values
    float sediment_change = arrays->_sediment_map_out[idx];
#else
    float water_outflow = 0.0f; // (⚠️ could precompute)
    float sediment_change = 0.0f;
#endif
    float water_inflow = 0.f; // needs to be calculated from neightbours

    for (int n = 0; n < 8; ++n) {

#ifndef EROSION_OUTFLOW_PRECALCULATION
        water_outflow += arrays->_flux8[idx * 8 + n];            // outflow from this tile (⚠️ could precompute)
        sediment_change -= arrays->_sediment_flux8[idx * 8 + n]; // outflow from this tile (⚠️ could precompute)
#endif

        int2 new_pos = wrap_or_clamp_index(pos + offsets[n], map_size, pars->wrap);
        int new_idx = pos_to_idx(new_pos, map_size.x);
        int opposite_offset = opposite_offset_refs[n];

        water_inflow += arrays->_flux8[new_idx * 8 + opposite_offset];             //  inflow from neighbouring tiles
        sediment_change += arrays->_sediment_flux8[new_idx * 8 + opposite_offset]; // inflow from neighbouring tiles
    }

    water -= water_outflow;
    water += water_inflow;
    sediment += sediment_change;

    float erosion = water_outflow * pars->erosion_rate;  // max possible erosion
    float available_erosion = height - pars->min_height; // limit erosion to available rock above min_height
    erosion = min(erosion, available_erosion);           // can't erode more than we have rock
    erosion = max(erosion, 0.0);                         // erosion can't be negative

    if (pars->_debug) {
        atomicAdd(&(debug->_debug_erosion_total), erosion);
    }

    sediment += erosion * pars->sediment_yield; // scale this by the material, some might make less sediment
    height -= erosion;

    // height = clamp(height, pars->min_height, pars->max_height); // ❓❓ pointless won't go up?

    if (pars->_debug) {
        float evaporation_loss = min(pars->evaporation_rate, water); // evaporation is either evaporation_rate or remaining
        atomicAdd(&(debug->_debug_evaporation_total), evaporation_loss);
    }
    water -= pars->evaporation_rate; // evaporation

    //

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
        if (pars->_debug) {

            // extra calculations to track drain
            float before = water;
            float drained = fminf(pars->drain_rate, before); // can't drain more than we have
            atomicAdd(&(debug->_debug_drain_total), drained);

            water -= pars->drain_rate; // are ignored in practise as we clip at end of process

            sediment -= pars->drain_rate * pars->sediment_capacity; // ⚠️ new drain away sediment to

        } else {
            water -= pars->drain_rate;
        }
    }

    // ================================================================
    // [output]
    // ----------------------------------------------------------------

    arrays->height_map[idx] = height;
    arrays->water_map[idx] = fmaxf(0.f, water);       // no negative water
    arrays->sediment_map[idx] = fmaxf(0.f, sediment); // no negative sediment
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

    if (pars._debug) {
        printf("⚠️  debug mode active!\n");
    }

    bool layers_mode = false;

    if (!layer_map.empty()) {
        printf("🐙 layer_map detected...\n");
        layers_mode = true;
        pars._width = layer_map.dimensions()[0];
        pars._height = layer_map.dimensions()[1];
        pars._layers = layer_map.dimensions()[2];
    } else if (!height_map.empty()) {
        printf("🐙 height_map detected...\n");
        pars._width = height_map.dimensions()[0];
        pars._height = height_map.dimensions()[1];
        pars._layers = 0;
    } else {
        throw std::runtime_error("layer_map and height_map empty!");
    }

    size_t array_size = pars._width * pars._height;

    _flux8.resize({array_size * 8}); // flux output
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

    // allocate arrays
#define ALLOCATE_ARRAYS \
    X(_water_map_out)   \
    X(_sediment_map_out)
#define X(NAME) \
    NAME.resize({pars._width, pars._height});
    ALLOCATE_ARRAYS
#undef X
#undef ALLOCATE_ARRAYS

    dev_array_ptrs.upload(get_array_ptrs());

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

    for (int step = 0; step < pars.steps; ++step) {

        calculate_flux2<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(),
            dev_array_ptrs.dev_ptr(),
            debug_outputs.dev_ptr(),
            step);

        apply_flux2<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(),
            dev_array_ptrs.dev_ptr(),
            debug_outputs.dev_ptr(),
            step);
    }

    stream.sync();
}

void TEMPLATE_CLASS_NAME::debug_update() {
    // pars = dev_pars.download();
    // stream.sync();
    // cudaDeviceSynchronize(); // required as we didn't implement stream in dev_pars

    debug_outputs.set_stream(stream.get());
    debug_outputs.download();
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
