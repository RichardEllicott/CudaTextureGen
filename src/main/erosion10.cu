/*

Dynamic behaviors:

Ice: erodes when warm/fast, grows when cold/slow.

Sand: high erosiveness, but can redeposit downstream.

Clay: erodes slowly, but once exposed it can slump catastrophically.

Bedrock: nearly immune, only erodes under extreme conditions.


*/

#include "core/cuda/curand_array_2d.cuh"
#include "erosion10.cuh"
// #include "erosion9_kernels.cuh"
// #include "noise_util.cuh"
#include "core.h" // timer
#include "cuda_math.cuh"
#include <stdexcept> // std::runtime_error
#include <stdint.h>

#define LAYER_ARRAY_LAYOUT 0

#define EROSION_OUTFLOW_PRECALCULATION

namespace TEMPLATE_NAMESPACE {

#pragma region CONSTANTS

constexpr float SQRT2 = 1.4142135623730950488f;      // root of 2
constexpr float INV_SQRT2 = 0.70710678118654752440f; // inverse root of 2
// constexpr float PI = 3.14159265358979323846f;
// constexpr float GOLDEN_RATIO = 1.6180339887498948482f;

// doesn't like this in cuda
// constexpr int2 OFFSETS[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // 8 offsets with the opposites in pairs, first 4 cardinal
// constexpr float OFFSET_DISTANCES[8] = {1.0f, 1.0f, 1.0f, 1.0f, SQRT2, SQRT2, SQRT2, SQRT2};
// constexpr int OFFSET_OPPOSITE_REFS[8] = {1, 0, 3, 2, 5, 4, 7, 6};
// constexpr float2 UNIT_OFFSETS_8[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {0.5, 0.5}, {-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}};

// GRID_OFFSETS_8 ??
__device__ __constant__ int2 OFFSETS[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // 8 offsets with the opposites in pairs, first 4 cardinal
__device__ __constant__ float OFFSET_DISTANCES[8] = {1.0f, 1.0f, 1.0f, 1.0f, SQRT2, SQRT2, SQRT2, SQRT2};
__device__ __constant__ int OFFSET_OPPOSITE_REFS[8] = {1, 0, 3, 2, 5, 4, 7, 6};
__device__ __constant__ float2 UNIT_OFFSETS_8[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {INV_SQRT2, INV_SQRT2}, {-INV_SQRT2, -INV_SQRT2}, {INV_SQRT2, -INV_SQRT2}, {-INV_SQRT2, INV_SQRT2}};

#pragma endregion

#pragma region HELPERS

// ping pong helper
template <typename T>
__device__ __forceinline__ T *get_map_ptr(T *in, T *out, int step) {
    return step % 2 == 0 ? in : out;
}

// ping pong helper
template <typename MapPtr>
__device__ __forceinline__ float read_map_in(MapPtr in, MapPtr out, int step, int idx) {
    return get_map_ptr(in, out, step)[idx];
}
// ping pong helper
template <typename MapPtr>
__device__ __forceinline__ void write_map_out(MapPtr in, MapPtr out, int step, int idx, float value) {
    get_map_ptr(in, out, step)[idx] = value;
}

#pragma endregion

#pragma region KERNELS2

__device__ inline float get_layered_height(
    const float *layer_map,
    const int2 map_size,
    const int layers,
    const int2 pos) {

    float height = 0.0;

    int idx = cuda_math::pos_to_idx(pos, map_size.x, layers);
    for (int layer = 0; layer < layers; ++layer) {
        height += layer_map[idx + layer];
    }

    return height;
}

// return exposed layer id, or if all layers empty return invalid ref == to total layers
__device__ inline int get_exposed_layer(
    const float *layer_map,
    const int layers,
    const int layer_idx // 2D idx * the layer count
) {
    int exposed_layer;
    for (int n = 0; n < layers; ++n) {
        float value = layer_map[layer_idx + n];
        if (value <= 0.0f) {
            exposed_layer = n + 1; // first exposed layer is next layer (possibly)
        } else {
            break; // layer is empty
        }
    }
    return exposed_layer;
}

__global__ void layer_mode_calculations3(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = cuda_math::pos_to_idx(pos, map_size.x);
    int layer_idx = idx * pars->_layers;
    // ================================================================
    // arrays->height_map[idx] = get_layered_height(arrays->layer_map, map_size, pars->_layers, pos);

#define FIND_EXPOSED_LAYER

    float height = 0.0;

    for (int n = 0; n < pars->_layers; ++n) {
        height += arrays->layer_map[layer_idx + n];
    }


}

#pragma endregion

#pragma region KERNELS3

__global__ void add_rain3(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = cuda_math::pos_to_idx(pos, map_size.x);
    // ================================================================
    // [Rain]
    // ----------------------------------------------------------------
    float rain = pars->rain_rate;
    if (arrays->rain_map) {
        rain *= arrays->rain_map[idx]; // multiply by rain_map if != nullptr
    }
    arrays->water_map[idx] += rain;
}

__global__ void calculate_flux3(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays,
    DebugOutputs *__restrict__ debug,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = cuda_math::pos_to_idx(pos, map_size.x);
    // ================================================================
    float *height_map = arrays->height_map;
    float *water_map = arrays->water_map;
    float *sediment_map = arrays->sediment_map;
    // ================================================================
    // float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];
    // float surface = height + water;
    // ================================================================
    // Calculate Slope Vector
    // ----------------------------------------------------------------

    int xp = cuda_math::wrap_or_clamp_index(pos.x + 1, map_size.x, pars->wrap); // x + 1
    int xn = cuda_math::wrap_or_clamp_index(pos.x - 1, map_size.x, pars->wrap); // x - 1
    int yp = cuda_math::wrap_or_clamp_index(pos.y + 1, map_size.y, pars->wrap); // y + 1
    int yn = cuda_math::wrap_or_clamp_index(pos.y - 1, map_size.y, pars->wrap); // y - 1

    int xp_idx = pos.y * map_size.x + xp; // {+1,0}
    int xn_idx = pos.y * map_size.x + xn; // {-1,0}
    int yp_idx = yp * map_size.x + pos.x; // {0,+1}
    int yn_idx = yn * map_size.x + pos.x; // {0,-1}

    // positive offsets data
    float xp_height = height_map[xp_idx];
    float yp_height = height_map[yp_idx];
    float xp_water = water_map[xp_idx];
    float yp_water = water_map[yp_idx];
    float xp_surface = xp_height + xp_water;
    float yp_surface = yp_height + yp_water;

    // negative offsets data
    float xn_height = height_map[xn_idx];
    float yn_height = height_map[yn_idx];
    float xn_water = water_map[xn_idx];
    float yn_water = water_map[yn_idx];
    float xn_surface = xn_height + xn_water;
    float yn_surface = yn_height + yn_water;
    // ----------------------------------------------------------------
    // optional jitter
    if (pars->slope_jitter) {
        switch (pars->slope_jitter_mode) {
        case 0: { // cheaper, reuses one hash, lower quality random shouldn't be a problem over frames
            uint32_t h = cuda_math::hash_uint(pos.x, pos.y, step, 0);
            xp_surface += cuda_math::hash_to_4randf(h, 0) * pars->slope_jitter;
            yp_surface += cuda_math::hash_to_4randf(h, 1) * pars->slope_jitter;
            xn_surface += cuda_math::hash_to_4randf(h, 2) * pars->slope_jitter;
            yn_surface += cuda_math::hash_to_4randf(h, 3) * pars->slope_jitter;
            break;
        }
        case 1: { // uses 4 hashes, technically better random
            xp_surface += cuda_math::hash_float_signed(pos.x, pos.y, step, 0) * pars->slope_jitter;
            yp_surface += cuda_math::hash_float_signed(pos.x, pos.y, step, 1) * pars->slope_jitter;
            xn_surface += cuda_math::hash_float_signed(pos.x, pos.y, step, 2) * pars->slope_jitter;
            yn_surface += cuda_math::hash_float_signed(pos.x, pos.y, step, 3) * pars->slope_jitter;
            break;
        }
        }
    }
    // ----------------------------------------------------------------
    float2 slope_vector = float2{xn_surface - xp_surface, yn_surface - yp_surface}; // note slope may be double actual (use scale to compensate)
    slope_vector /= pars->scale;                                                    // scale such that double world size would mean half gradients
    // ----------------------------------------------------------------

    int idx2 = idx * 2;
    arrays->_slope_vector2[idx2] = slope_vector.x; // save to a vector map for later use
    arrays->_slope_vector2[idx2 + 1] = slope_vector.y;

    float slope_magnitude = cuda_math::length(slope_vector); // 🧪 save magnitude (OPTIONAL)
    arrays->_slope_magnitude[idx] = slope_magnitude;

    float water_velocity = pow(water, 2.0f / 3.0f) * sqrt(slope_magnitude); // 🧪 manning based velocity??
    arrays->_water_velocity[idx] = water_velocity;

    // ================================================================
    // Calculate Fluxes
    // ----------------------------------------------------------------

    // these offsets scale the result of the diagonals down by using a magnitude of 1/SQRT(2), the axis of this is 0.5
    constexpr float2 DOT_OFFSETS_8[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {0.5, 0.5}, {-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}};

    float fluxes[8];
    float flux_total = 0.0f;
    for (int n = 0; n < 8; ++n) {
        float2 unit_offset = DOT_OFFSETS_8[n];                     // cell offset as a float vector, not normalized as this helps scale
        float product = cuda_math::dot(unit_offset, slope_vector); // dot product gets how strongly we push into this direction
        product = max(product, 0.0);                               // positive only
        fluxes[n] = product;
        flux_total += product;
    }

    // normalize (all add up to 1.0)
    if (flux_total > 1e-6f) {
        for (int n = 0; n < 8; ++n) { fluxes[n] /= flux_total; }
    } else {
        for (int n = 0; n < 8; ++n) { fluxes[n] = 0.0f; } // prevents div by zero (just set 0)
    }

    float water_outflow = flux_total * pars->flow_rate;          // water outflow total
    water_outflow = min(water_outflow, water);                   // can't exceed water
    water_outflow = min(water_outflow, pars->max_water_outflow); // or max outflow

    float sediment_outflow = water_outflow * pars->sediment_capacity; // sediment outflow total
    sediment_outflow = min(sediment_outflow, sediment);               // can't exceed sediment

    int idx8 = idx * 8;
    float *_flux8_ptr = &arrays->_flux8[idx8];
    float *_sediment_flux8_ptr = &arrays->_sediment_flux8[idx8];
    for (int n = 0; n < 8; ++n) {
        float flux = fluxes[n];
        _flux8_ptr[n] = flux * water_outflow;
        _sediment_flux8_ptr[n] = flux * sediment_outflow;
    }

    // save the total outflow's rather than add them up again later
    arrays->_water_out[idx] = water_outflow;
    arrays->_sediment_out[idx] = sediment_outflow;
}

__global__ void apply_flux3(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays,
    DebugOutputs *__restrict__ debug,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = cuda_math::pos_to_idx(pos, map_size.x);
    // int idx8 = idx * 8;
    // ================================================================
    float *height_map = arrays->height_map;
    float *water_map = arrays->water_map;
    float *sediment_map = arrays->sediment_map;
    // ================================================================
    float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];
    // float surface = height + water;
    // ================================================================
    // Find Exposed Layer (if layer mode active)
    // ----------------------------------------------------------------
    bool layer_mode = pars->_layers > 0; // if layer mode active
    int exposed_layer;
    int layer_idx;
    if (layer_mode) {
        layer_idx = idx * pars->_layers;
        exposed_layer = get_exposed_layer(arrays->layer_map, pars->_layers, layer_idx);
    }
    // ================================================================
    // Flow
    // ----------------------------------------------------------------
    float water_out = arrays->_water_out[idx];           // water out already calculated
    float sediment_change = -arrays->_sediment_out[idx]; // sediment out already calculated
    float water_in = 0.0f;                               // to calculate from neighbours

    for (int n = 0; n < 8; ++n) {
        int2 new_pos = cuda_math::wrap_or_clamp_index(pos + OFFSETS[n], map_size, pars->wrap);
        int new_idx = cuda_math::pos_to_idx(new_pos, map_size.x);
        int new_idx8 = new_idx * 8;
        int opposite_ref = OFFSET_OPPOSITE_REFS[n];

        water_in += arrays->_flux8[new_idx8 + opposite_ref]; //  inflow from neighbouring tiles
        sediment_change += arrays->_sediment_flux8[new_idx8 + opposite_ref];
    }

    water += water_in;
    water -= water_out;

    // ================================================================
    // Erosion
    // ----------------------------------------------------------------
    float available_erosion = height - pars->min_height; // limit erosion to available rock above min_height
    float erosion;
    switch (pars->erosion_mode) {
    case 0: // simple water * erosion_rate
        erosion = water_out * pars->erosion_rate;
        break;
    case 1:
        erosion = water_out * pars->erosion_rate * arrays->_slope_magnitude[idx]; // with slope_magnitude ⚠️ BROKE?
        break;
    case 2:
        erosion = water * pars->erosion_rate * arrays->_slope_magnitude[idx]; // maybe based on total water?
        break;
    case 3:
        erosion = water * pars->erosion_rate * arrays->_water_velocity[idx]; // maybe based on total water?
        break;
    case 4: // soft saturation scheme (limits the max erosion)
        erosion = cuda_math::soft_saturate(arrays->_water_velocity[idx], pars->erosion_rate, 1.0);
        break;
    }

    erosion = min(erosion, available_erosion); // can't erode more than we have rock
    erosion = max(erosion, 0.0f);              // erosion can't be negative

    //
    //
    //
    // note this new thing:
    // #pragma unroll

    // we could even concider runtime compile of stuff!
    // https://copilot.microsoft.com/chats/UnoMio7MXZLWKrCQAQzed
    // we may precompute this exposed layer

    // ================================================================

    // get_exposed_layer

    if (layer_mode) {
        if (exposed_layer < pars->_layers) { // if layers not empty
            float layer_erosiveness = arrays->layer_erosiveness[exposed_layer];
            float layer_height = arrays->layer_map[layer_idx + exposed_layer];
            layer_height -= erosion * layer_erosiveness; // use actual layer properties
            layer_height = max(layer_height, 0.0f);      // no lower than 0.0
            arrays->layer_map[layer_idx + exposed_layer] = layer_height;
        }

    } else {
        height -= erosion; // normal mode
    }

    if (pars->sediment_layer_mode) {

    } else {
        sediment += erosion * pars->sediment_yield;
    }

    // ================================================================
    // [Deposition]
    // ----------------------------------------------------------------
    if (water_out < pars->deposition_threshold) {
        float deposit = sediment * pars->deposition_rate;
        sediment -= deposit;
        height += deposit;
    }
    // ================================================================
    // Drain
    // ----------------------------------------------------------------
    if (pars->drain_rate > 0.0f && height <= pars->min_height) {
        water -= pars->drain_rate; // minus drain (could be negative)
    }
    // ================================================================
    // Evaporation
    // ----------------------------------------------------------------
    switch (pars->evaporation_mode) {
    case 0:
        water -= pars->evaporation_rate; // normal (could be negative)
        break;
    case 1:
        break;
    }
    // ================================================================
    // [Output]
    // ----------------------------------------------------------------
    height = cuda_math::clamp(height, pars->min_height, pars->max_height);
    water = max(water, 0.0f); // prevent negative water
    sediment = max(sediment, 0.0f);

    height_map[idx] = height;
    water_map[idx] = water;
    sediment_map[idx] = sediment;
}

__global__ void sea_pass3(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays,
    DebugOutputs *__restrict__ debug,
    const int step) {
}

#pragma endregion

void TEMPLATE_CLASS_NAME::allocate_device() {

    if (_device_allocated) { return; }

    if (pars._debug) { printf("⚠️  debug mode active!\n"); }

    if (!layer_map.empty()) {
        printf("🐡 layer_map detected...\n");
        pars._width = layer_map.dimensions()[0];
        pars._height = layer_map.dimensions()[1];
        pars._layers = layer_map.dimensions()[2]; // marks layers mode as active

        height_map.set_stream(stream.get());
        height_map.resize({pars._width, pars._height}); // allocate the heightmap still, we will copy the total height to it

        if (pars.sediment_layer_mode) {
            sediment_layer_map.resize(layer_map.dimensions());
        }

        _exposed_layer_map.resize(height_map.dimensions()); // set _exposed_layer_map to height_map dimensions



        // ensure all arrays have 1's
        std::vector<float> ones(pars._layers, 1.0f); // vector of 1.0's
#define LAYER_DATA_ARRAYS      \
    X(layer_erosiveness)       \
    X(layer_yield)             \
    X(layer_permeability)      \
    X(layer_erosion_threshold) \
    X(layer_solubility)

#define X(NAME)                                   \
    if (NAME.size() < pars._layers) {             \
        NAME.resize({pars._layers});              \
        NAME.upload(ones.data(), {pars._layers}); \
    }
        LAYER_DATA_ARRAYS
#undef X
        stream.sync(); // ensure we don't free the vector before upload
#undef LAYER_DATA_ARRAYS

        // stream.sync();
    } else if (!height_map.empty()) {
        printf("🐡 height_map detected...\n");
        pars._width = height_map.dimensions()[0];
        pars._height = height_map.dimensions()[1];
        // pars._layers = 1;
    } else {
        throw std::runtime_error("layer_map and height_map empty!");
    }

#define DEBUGGING_LAYERS
#ifdef DEBUGGING_LAYERS
    // will move to main loop eventually
    stream.sync();
    configure_device();
    stream.sync();
#endif

    size_t array_size = pars._width * pars._height;

    _flux8.resize({array_size * 8});                        // flux output
    _sediment_flux8.resize({array_size * 8});               // sediment flux output
    _slope_vector2.resize({pars._width, pars._height, 2}); // 2D vectors



// allocate and zero arrays
#define ZERO_ARRAYS \
    X(water_map)    \
    X(sediment_map)
#define X(NAME)                               \
    if (NAME.empty()) {                       \
        NAME.resize(height_map.dimensions()); \
        NAME.zero_device();                   \
    }
    ZERO_ARRAYS
#undef X
#undef ZERO_ARRAYS

// allocate arrays
#define ALLOCATE_ARRAYS \
    X(_slope_magnitude) \
    X(_water_out)       \
    X(_sediment_out)    \
    X(_water_velocity)
#define X(NAME) \
    NAME.resize(height_map.dimensions());
    ALLOCATE_ARRAYS
#undef X
#undef ALLOCATE_ARRAYS

    dev_array_ptrs.upload(get_array_ptrs());

#ifdef DEBUGGING_LAYERS
    if (pars._layers > 0) {

        layer_mode_calculations3<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(),
            dev_array_ptrs.dev_ptr());
    };
#endif
#undef DEBUGGING_LAYERS

    _device_allocated = true;
} // namespace TEMPLATE_NAMESPACE

void TEMPLATE_CLASS_NAME::process() {

    allocate_device();
    configure_device();
    stream.sync();

    for (int i = 0; i < pars.steps; ++i) {

        if (pars._layers > 0) {
            layer_mode_calculations3<<<grid, block, 0, stream.get()>>>(
                dev_pars.dev_ptr(),
                dev_array_ptrs.dev_ptr());
        }

        add_rain3<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(),
            dev_array_ptrs.dev_ptr());

        calculate_flux3<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(),
            dev_array_ptrs.dev_ptr(),
            debug_outputs.dev_ptr(),
            pars._step);

        apply_flux3<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(),
            dev_array_ptrs.dev_ptr(),
            debug_outputs.dev_ptr(),
            pars._step);

        pars._step++;
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
