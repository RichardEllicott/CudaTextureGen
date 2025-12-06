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

__global__ void add_rain2(
    Parameters *pars,
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
    if (arrays->rain_map) {
        rain *= arrays->rain_map[idx]; // multiply by rain_map if != nullptr
    }
    arrays->water_map[idx] += rain;

    if (pars->debug) {
        // __shared__ float block_sum;
        // block_sum = 0.0f;
        // __syncthreads();

        // float delta = rain;
        // atomicAdd(&block_sum, delta);

        // __syncthreads();
        // if (threadIdx.x == 0) {
        //     atomicAdd(total, block_sum);
        // }

        atomicAdd(&(pars->_debug_rain_total), rain);
    }
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

        float new_height = arrays->height_map[new_idx];
        float new_water = arrays->water_map[new_idx];
        float new_sediment = arrays->sediment_map[new_idx];

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
    float height = arrays->height_map[idx];
    float water = arrays->water_map[idx];
    float sediment = arrays->sediment_map[idx];
    // ================================================================
    // [Calculate Flows]
    // ----------------------------------------------------------------

    float water_inflow = 0.f;     // inflow calculated by visting neighbours
    float sediment_change = 0.0f; // (⚠️ could partially precompute)

    float water_outflow = 0.0f; // (⚠️ could precompute)

    for (int n = 0; n < 8; ++n) {
        water_outflow += arrays->_flux8[idx * 8 + n];            // outflow from this tile (⚠️ could precompute)
        sediment_change -= arrays->_sediment_flux8[idx * 8 + n]; // outflow from this tile (⚠️ could precompute)

        int2 new_pos = wrap_or_clamp_index(pos + offsets[n], map_size, pars->wrap);
        int new_idx = pos_to_idx(new_pos, map_size.x);
        int opposite_offset = opposite_offset_refs[n];

        water_inflow += arrays->_flux8[new_idx * 8 + opposite_offset];             //  inflow from neighbouring tiles
        sediment_change += arrays->_sediment_flux8[new_idx * 8 + opposite_offset]; // inflow from neighbouring tiles
    }

    water -= water_outflow;
    water += water_inflow;
    sediment += sediment_change;

    float erosion = water_outflow * pars->erosion_rate;
    float available_erosion = height - pars->min_height; // limit erosion to available rock above min_height
    erosion = fminf(erosion, fmaxf(0.0f, available_erosion));

    sediment += erosion * pars->sediment_yield; // ❓ scale this by the material, some might make less sediment
    height -= erosion;

    height = clamp(height, pars->min_height, pars->max_height); // ❓❓ pointless won't go up?

    water -= pars->evaporation_rate; // evaporation

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

    if (pars.debug) {
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

        // add_rain2<<<grid, block, 0, stream.get()>>>(dev_pars.dev_ptr(), dev_array_ptrs.dev_ptr(), step);
        calculate_flux2<<<grid, block, 0, stream.get()>>>(dev_pars.dev_ptr(), dev_array_ptrs.dev_ptr(), step);
        apply_flux2<<<grid, block, 0, stream.get()>>>(dev_pars.dev_ptr(), dev_array_ptrs.dev_ptr(), step);
    }

    stream.sync();
}

void TEMPLATE_CLASS_NAME::debug_update() {
    pars = dev_pars.download();
    stream.sync();
    cudaDeviceSynchronize(); // required as we didn't implement stream in dev_pars
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
