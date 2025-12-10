#include "core/cuda/curand_array_2d.cuh"
#include "erosion10.cuh"
// #include "erosion9_kernels.cuh"
// #include "noise_util.cuh"
#include "core.h" // timer
#include "cuda_math.cuh"
#include <stdexcept> // std::runtime_error
#include <stdint.h>

// #define EROSION_OUTFLOW_PRECALCULATION

namespace TEMPLATE_NAMESPACE {

#pragma region HASH

// integer hash (based on MurmurHash3 finalizer)
__device__ __forceinline__ int hash_int(int x, int y, int z, int seed) {
    int n = x + y * 374761393 + z * 668265263 + seed * 1274126177;

    n ^= n >> 16;
    n *= 0x85ebca6b;
    n ^= n >> 13;
    n *= 0xc2b2ae35;
    n ^= n >> 16;

    return n; // can be negative
}

// integer hash (based on MurmurHash3 finalizer)
__device__ __forceinline__ uint32_t hash_uint(uint32_t x, uint32_t y, uint32_t z, uint32_t seed) {
    uint32_t n = x + y * 374761393u + z * 668265263u + seed * 1274126177u;

    n ^= n >> 16;
    n *= 0x85ebca6bu;
    n ^= n >> 13;
    n *= 0xc2b2ae35u;
    n ^= n >> 16;

    return n; // full 32-bit unsigned result
}

// float from [0,1]
__device__ __forceinline__ float hash_float(uint32_t hash) {
    return static_cast<float>(hash) / pow(2.0f, 32.0f); // Scale to [0,1]
}
// float from [0,1]
__device__ __forceinline__ float hash_float(uint32_t x, uint32_t y, uint32_t z, uint32_t seed) {
    return hash_float(hash_uint(x, y, z, seed));
}

// float from [-1,1]
__device__ __forceinline__ float hash_float_signed(int hash) {
    return static_cast<float>(hash) / pow(2.0f, 31.0f); // Scale to [-1,1).
}

// float from [-1,1] range:
__device__ __forceinline__ float hash_float_signed(int x, int y, int z, int seed) {
    return hash_float_signed(hash_int(x, y, z, seed));
}

// take in a hash, extract a bool (set index from 0 to 31)
__device__ __forceinline__ bool hash_to_bool(int hash, int index = 0) {
    return (hash >> index) & 1u;
}

// if ((hash_result >> 0) & 1u) { // we can extract from 0-31 for a 50:50
// }

#pragma endregion

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

__device__ __forceinline__ int pos_to_idx(int2 pos, int map_width) {
    return pos.y * map_width + pos.x;
}

__device__ __forceinline__ int pos_to_idx(int x, int y, int map_width) {
    return y * map_width + x;
}

#pragma endregion

#pragma region KERNELS2

__device__ inline float get_layered_height(
    const float *layer_map,
    const int map_width,
    const int layers,
    const int2 pos) {

    float height = 0.0;
    int idx = (pos.y * map_width + pos.x) * layers;
    for (int n = 0; n < layers; ++n) {
        height += layer_map[idx + n];
    }

    return height;
}

__global__ void precalc_layer_height(
    const Parameters *pars,
    const ArrayPtrs *arrays) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = pos_to_idx(pos, map_size.x);
    // ================================================================
    arrays->height_map[idx] = get_layered_height(arrays->layer_map, pos.x, pars->_layers, pos);
}

#pragma endregion

#pragma region KERNELS3

// could inline
// __global__ void add_rain3(
//     const Parameters *pars,
//     const ArrayPtrs *arrays) {

// __restrict__ here, might be good it promises only one thing points to this, so it's not added twice

__global__ void add_rain3(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays) {
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
    int idx = pos_to_idx(pos, map_size.x);
    // ================================================================
    float *height_map = arrays->height_map;
    float *water_map = arrays->water_map;
    float *sediment_map = arrays->sediment_map;
    // ================================================================
    float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];
    float surface = height + water;
    // ================================================================
    // HASH TEST
    // int hash = hash_int(pos.x, pos.y, step, 0);
    // if (hash_to_bool(hash)){

    // }
    // ================================================================
    // optional jitter added by offsetting surface by random value
    if (pars->slope_jitter > 0.0f) {
        switch (pars->slope_jitter_mode) {
        case 0:
            surface += hash_float_signed(pos.x, pos.y, step, 1) * pars->slope_jitter; // change each step
            break;
        default:
            surface += hash_float_signed(pos.x, pos.y, 0, 1) * pars->slope_jitter; // same each step (probabally doesn't really do anything)
            break;
        }
    }
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

    float2 slope_vector = float2{xn_surface - xp_surface, yn_surface - yp_surface}; // note slope may be double actual (use scale to compensate)

    slope_vector /= pars->scale; // scale such that double world size would mean half gradients

    // save to a Flow map for later use
    int idx2 = idx * 2;
    arrays->_slope_vector2[idx2] = slope_vector.x;
    arrays->_slope_vector2[idx2 + 1] = slope_vector.y;

    float slope_magnitude = cuda_math::length(slope_vector);
    arrays->_slope_magnitude[idx] = slope_magnitude;

// ================================================================
// Calculate Fluxes
// ----------------------------------------------------------------
// #define BAKE_SCALE_TO_UNIT_OFFSETS_8
// #ifdef BAKE_SCALE_TO_UNIT_OFFSETS_8
//     constexpr float2 UNIT_OFFSETS_8[8] = {
//         {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {0.5, 0.5}, {-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}}; // baking scale here penalizes the dot product result of diagonals
// #endif

    float fluxes[8];
    float flux_total = 0.0f;
    for (int n = 0; n < 8; ++n) {
        float2 unit_offset = UNIT_OFFSETS_8[n];                    // cell offset as a float vector, not normalized as this helps scale
        float product = cuda_math::dot(unit_offset, slope_vector); // dot product gets how strongly we push into this direction
        product = max(product, 0.0);                               // positive only
        fluxes[n] = product;
        flux_total += product;
    }
    // ----------------------------------------------------------------
    // // normalize (all add up to 1.0)
    // if (flux_total > 1e-6f) {
    //     for (int n = 0; n < 8; ++n) {
    //         fluxes[n] /= flux_total;
    //     }
    // } else {
    //     for (int n = 0; n < 8; ++n) {
    //         fluxes[n] = 0.0f;
    //     }
    // }
    // ================================================================
    // Outflows
    // ----------------------------------------------------------------

    float flux_scale = 1.0f;                                        /// value to scale flux, never up but only down
    float max_or_total_water = min(water, pars->max_water_outflow); // the total water or the max outflow
    if (flux_total > 1e-6f) {
        if (flux_total > max_or_total_water) {
            flux_scale = max_or_total_water / flux_total; // this will only scale down
        }
    } else {
        flux_scale = 0.0f;
    }

    // ----------------------------------------------------------------

    int idx8 = idx * 8;
    float *_flux8_ptr = &arrays->_flux8[idx8];
    float *_sediment_flux8_ptr = &arrays->_sediment_flux8[idx8];
    for (int n = 0; n < 8; ++n) {
        _flux8_ptr[n] = fluxes[n] * flux_scale;
    }
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
    int idx = pos_to_idx(pos, map_size.x);
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
    // Flow
    // ----------------------------------------------------------------

    float water_out = 0.0f;
    float water_in = 0.0f;

    int idx8 = idx * 8;

    for (int n = 0; n < 8; ++n) {
    }

    for (int n = 0; n < 8; ++n) {

        water_out += arrays->_flux8[idx8 + n]; //  (⚠️ could precompute)

        int2 new_pos = cuda_math::wrap_or_clamp_index(pos + OFFSETS[n], map_size, pars->wrap);
        int new_idx = pos_to_idx(new_pos, map_size.x);
        int opposite_ref = OFFSET_OPPOSITE_REFS[n];

        water_in += arrays->_flux8[new_idx * 8 + opposite_ref]; //  inflow from neighbouring tiles
    }

    water += water_in;
    water -= water_out;

    // ================================================================
    // Evaporation
    // ----------------------------------------------------------------
    water -= pars->evaporation_rate;

    //
    // plain outflow
    //
    float erosion = water_out * pars->erosion_rate;      // max possible erosion
    float available_erosion = height - pars->min_height; // limit erosion to available rock above min_height
    erosion = min(erosion, available_erosion);           // can't erode more than we have rock
    erosion = max(erosion, 0.0f);                        // erosion can't be negative

    //
    // outflow * slope  ⚠️ BROKE
    //
    // float slope_magnitude = arrays->_slope_magnitude[idx];
    // float erosion = water_out * pars->erosion_rate * slope_magnitude; // max possible erosion
    // float available_erosion = height - pars->min_height;              // limit erosion to available rock above min_height
    // erosion = min(erosion, available_erosion);                        // can't erode more than we have rock
    // erosion = max(erosion, 0.0f);                                     // erosion can't be negative
    //
    //
    //
    height -= erosion;
    sediment += erosion;

    // ================================================================
    // Drain
    // ----------------------------------------------------------------
    if (pars->drain_rate > 0.0f && height <= pars->min_height) {
        water -= pars->drain_rate;
    }

    // ================================================================
    water = max(water, 0.0f);
    height_map[idx] = height;
    water_map[idx] = water;
    sediment_map[idx] = sediment;
}

#pragma endregion

void TEMPLATE_CLASS_NAME::allocate_device() {

    if (_device_allocated)
        return;

    if (pars._debug) {
        printf("⚠️  debug mode active!\n");
    }

    if (!layer_map.empty()) {
        printf("🐡 layer_map detected...\n");
        pars._width = layer_map.dimensions()[0];
        pars._height = layer_map.dimensions()[1];
        pars._layers = layer_map.dimensions()[2]; // marks layers mode as active

    } else if (!height_map.empty()) {
        printf("🐡 height_map detected...\n");
        pars._width = height_map.dimensions()[0];
        pars._height = height_map.dimensions()[1];
        pars._layers = 0;
    } else {
        throw std::runtime_error("layer_map and height_map empty!");
    }

    size_t array_size = pars._width * pars._height;

    _flux8.resize({array_size * 8});                        // flux output
    _sediment_flux8.resize({array_size * 8});               // sediment flux output
    _slope_vector2.resize({pars._width * 2, pars._height}); // double size

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
    X(_slope_magnitude)
#define X(NAME) \
    NAME.resize({pars._width, pars._height});
    ALLOCATE_ARRAYS
#undef X
#undef ALLOCATE_ARRAYS

    dev_array_ptrs.upload(get_array_ptrs());

    _device_allocated = true;
}

void TEMPLATE_CLASS_NAME::process() {

    allocate_device();
    configure_device();
    stream.sync();

    for (int step = 0; step < pars.steps; ++step) {

        add_rain3<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(),
            dev_array_ptrs.dev_ptr());

        // if (pars._layers > 0) {
        //     precalc_layer_height<<<grid, block, 0, stream.get()>>>(
        //         dev_pars.dev_ptr(),
        //         dev_array_ptrs.dev_ptr());
        // };

        calculate_flux3<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(),
            dev_array_ptrs.dev_ptr(),
            debug_outputs.dev_ptr(),
            step);

        apply_flux3<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(),
            dev_array_ptrs.dev_ptr(),
            debug_outputs.dev_ptr(),
            step);
    }

    // stream.sync();
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
