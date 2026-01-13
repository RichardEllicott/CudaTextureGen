/*

Dynamic behaviors:

Ice: erodes when warm/fast, grows when cold/slow.

Sand: high erosiveness, but can redeposit downstream.

Clay: erodes slowly, but once exposed it can slump catastrophically.

Bedrock: nearly immune, only erodes under extreme conditions.


*/
#define PRECALCULATE_EXPOSED_LAYER
#define LAYER_ARRAY_LAYOUT 0

#include "core/cuda/curand_array_2d.cuh"
#include "erosion10.cuh"
// #include "erosion9_kernels.cuh"
// #include "noise_util.cuh"
#include "core.h" // timer
#include "core/cuda/math.cuh"
#include "core/cuda/operators.cuh"
#include <stdexcept> // std::runtime_error
#include <stdint.h>

#include <cuda_runtime.h>
// #include <cuda_math_constants.h>

#define EROSION_OUTFLOW_PRECALCULATION

namespace TEMPLATE_NAMESPACE {

// using namespace core::cuda::math;
namespace math = core::cuda::math;

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
DH_INLINE T *get_map_ptr(T *in, T *out, int step) {
    return step % 2 == 0 ? in : out;
}

// ping pong helper
template <typename MapPtr>
DH_INLINE float read_map_in(MapPtr in, MapPtr out, int step, int idx) {
    return get_map_ptr(in, out, step)[idx];
}
// ping pong helper
template <typename MapPtr>
DH_INLINE void write_map_out(MapPtr in, MapPtr out, int step, int idx, float value) {
    get_map_ptr(in, out, step)[idx] = value;
}

// get total height of ground iterating the layers
DH_INLINE float get_layered_height(
    const float *layer_map,
    const int layers,
    const int layer_idx) {

    float height = 0.0;
    for (int n = 0; n < layers; ++n) {
        height += layer_map[layer_idx + n];
    }
    return height;
}

// return exposed layer id, or if all layers empty return invalid ref == to total layers
DH_INLINE int get_exposed_layer(
    const float *__restrict__ layer_map,
    const int layer_count,
    const int layer_idx // 2D idx * the layer count
) {
    int exposed_layer;
    for (int n = 0; n < layer_count; ++n) {
        float value = layer_map[layer_idx + n];
        if (value <= 0.0f) {
            exposed_layer = n + 1; // first exposed layer is next layer (possibly)
        } else {
            break; // layer is empty
        }
    }
    return exposed_layer;
}

#pragma endregion

#pragma region KERNELS3

// calculate the height based on layers and get exposed layer
__global__ void layer_mode_calculations3(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = math::pos_to_idx(pos, map_size.x);
    int layer_idx = idx * pars->_layers;
    // ================================================================

    arrays->height_map[idx] = get_layered_height(arrays->layer_map, pars->_layers, layer_idx);

#ifdef PRECALCULATE_EXPOSED_LAYER
    int exposed_layer = get_exposed_layer(arrays->layer_map, pars->_layers, layer_idx);
    arrays->_exposed_layer_map[idx] = exposed_layer;
#endif
}

// add rain
// requires:
// water_map
// rain_map (optional)
__global__ void add_rain3(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = math::pos_to_idx(pos, map_size.x);
    // ================================================================
    // [Rain]
    // ----------------------------------------------------------------
    float rain = pars->rain_rate;
    if (arrays->rain_map) {
        rain *= arrays->rain_map[idx]; // multiply by rain_map if != nullptr
    }
    arrays->water_map[idx] += rain;
}

// calculate water and sediment flux (or total outflow)
// requires:
// height_map
// water_map
// sediment_map
// _slope_vector2 ❓ UNUSED so far
// _slope_magnitude ❓ USED FOR SOME EROSION MODES
// _water_velocity ❓USED FOR SOME EROSION MODES
// _flux8
// _sediment_flux8
// _water_out ❗ avoids recalculating
// _sediment_out ❗ avoids recalculating
__global__ void calculate_outflow3(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays,
    DebugOutputs *__restrict__ debug,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = math::pos_to_idx(pos, map_size.x);
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

    int xp = math::wrap_or_clamp_index(pos.x + 1, map_size.x, pars->wrap); // x + 1
    int xn = math::wrap_or_clamp_index(pos.x - 1, map_size.x, pars->wrap); // x - 1
    int yp = math::wrap_or_clamp_index(pos.y + 1, map_size.y, pars->wrap); // y + 1
    int yn = math::wrap_or_clamp_index(pos.y - 1, map_size.y, pars->wrap); // y - 1

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
            uint32_t h = math::hash_uint(pos.x, pos.y, step, 0);
            xp_surface += math::hash_to_4randf(h, 0) * pars->slope_jitter;
            yp_surface += math::hash_to_4randf(h, 1) * pars->slope_jitter;
            xn_surface += math::hash_to_4randf(h, 2) * pars->slope_jitter;
            yn_surface += math::hash_to_4randf(h, 3) * pars->slope_jitter;
            break;
        }
        case 1: { // uses 4 hashes, technically better random
            xp_surface += math::hash_float_signed(pos.x, pos.y, step, 0) * pars->slope_jitter;
            yp_surface += math::hash_float_signed(pos.x, pos.y, step, 1) * pars->slope_jitter;
            xn_surface += math::hash_float_signed(pos.x, pos.y, step, 2) * pars->slope_jitter;
            yn_surface += math::hash_float_signed(pos.x, pos.y, step, 3) * pars->slope_jitter;
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

    float slope_magnitude = math::length(slope_vector); // 🧪 save magnitude (OPTIONAL)
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
        float2 unit_offset = DOT_OFFSETS_8[n];                            // cell offset as a float vector, not normalized as this helps scale
        float product = math::dot(unit_offset, slope_vector); // dot product gets how strongly we push into this direction
        product = max(product, 0.0);                                      // positive only
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

// apply flows and erosion etc
// requires:
// height_map
// water_map
// sediment_map
// layer_map ❓ optional for layer mode
// _exposed_layer_map
// _slope_magnitude
// _water_velocity
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
    int idx = math::pos_to_idx(pos, map_size.x);
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

#ifdef PRECALCULATE_EXPOSED_LAYER
        exposed_layer = arrays->_exposed_layer_map[idx];
#else
        exposed_layer = get_exposed_layer(arrays->layer_map, pars->_layers, layer_idx);
#endif
    }
    // ================================================================
    // [Flow]
    // ----------------------------------------------------------------
    float water_out = arrays->_water_out[idx];           // water out already calculated
    float sediment_change = -arrays->_sediment_out[idx]; // sediment out already calculated
    float water_in = 0.0f;                               // to calculate from neighbours

    for (int n = 0; n < 8; ++n) {
        int2 new_pos = math::wrap_or_clamp_index(pos + OFFSETS[n], map_size, pars->wrap);
        int new_idx = math::pos_to_idx(new_pos, map_size.x);
        int new_idx8 = new_idx * 8;
        int opposite_ref = OFFSET_OPPOSITE_REFS[n];

        water_in += arrays->_flux8[new_idx8 + opposite_ref]; //  inflow from neighbouring tiles
        sediment_change += arrays->_sediment_flux8[new_idx8 + opposite_ref];
    }

    water += water_in;
    water -= water_out;

    // ================================================================
    // [Erosion]
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
        erosion = water * pars->erosion_rate * arrays->_water_velocity[idx]; // total water and the water velocity (manning)
        break;
    case 4: // soft saturation scheme (limits the max erosion)
        erosion = math::soft_saturate(arrays->_water_velocity[idx], pars->erosion_rate, 1.0);
        break;
    }

    // apply layer erosiveness after the erosion calculation (seems best if we use soft_saturate)
    if (layer_mode) {
        float layer_erosiveness = arrays->layer_erosiveness[exposed_layer];
        erosion *= layer_erosiveness;
    }

    erosion = math::clamp(erosion, 0.0f, available_erosion); // ensure not negative and not more than available_erosion

    // ================================================================
    // [Apply Erosion to Height]
    // ----------------------------------------------------------------
    if (layer_mode) {
        if (exposed_layer < pars->_layers) { // if layers not empty

            float layer_height = arrays->layer_map[layer_idx + exposed_layer];
            layer_height -= erosion;
            layer_height = max(layer_height, 0.0f); // no lower than 0.0
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

        if (layer_mode) {
            if (pars->sediment_layer_mode) {

            } else {
            }

        } else {
            height += deposit;
        }
    }
    // ================================================================
    // [Drain]
    // ----------------------------------------------------------------
    if (pars->drain_rate > 0.0f && height <= pars->min_height) {
        water -= pars->drain_rate; // minus drain (could be negative)
    }
    // ================================================================
    // [Evaporation]
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

    if (layer_mode) {
        // already applied height to layer
    } else {
        height_map[idx] = math::clamp(height, pars->min_height, pars->max_height);
    }
    water_map[idx] = max(water, 0.0f);
    sediment_map[idx] = max(sediment, 0.0f);
}

// 🚧 🚧 🚧 🚧 UNFINISHED
DH_INLINE float sea_fade_sine(float height, float avg_sea, float tide_range) {
    float half = 0.5f * tide_range;
    float low = avg_sea - half;
    float high = avg_sea + half;

    if (height <= low) return 1.0f;  // always submerged
    if (height >= high) return 0.0f; // never submerged

    float rel = (height - low) / tide_range;                             // 0..1
    float fade = 1.0f - acosf(1.0f - 2.0f * rel) / math::PI; // sine exposure fraction
    return fade;
}

__global__ void sea_pass3(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays,
    DebugOutputs *__restrict__ debug,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = math::pos_to_idx(pos, map_size.x);
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
    // [Tidal Fade]
    // ----------------------------------------------------------------
    float sea_fade = sea_fade_sine(height, pars->sea_level, pars->sea_tidal_range);
    arrays->_sea_map[idx] = sea_fade;

    // ================================================================
    height_map[idx] = height;
    water_map[idx] = water;
    sediment_map[idx] = sediment;
}

// simple erosion collapse
__global__ void simple_collapse4(
    const Parameters *__restrict__ pars,
    float *height_map,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = math::pos_to_idx(pos, map_size.x);
    // ================================================================
    const int offset_count = 4;
    float slope_threshold = pars->simple_collapse_threshold;
    float erosion_rate = pars->simple_collapse_amount;
    // float min_height = pars->min_height;
    float erosion_yield = pars->simple_collapse_yield;
    float jitter = pars->simple_collapse_jitter;

    float height = height_map[idx];
    if (jitter > 0.0f) {
        height += math::hash_float_signed(pos.x, pos.y, step, 482) * jitter;
    }

    // ================================================================
    // [Calculate]
    // ----------------------------------------------------------------
    float total_slope = 0.0f;
    float slopes[offset_count];
    int idxs[offset_count];
    for (int n = 0; n < offset_count; ++n) {
        int2 new_pos = math::wrap_or_clamp_index(pos + OFFSETS[n], map_size, pars->wrap);
        int new_idx = math::pos_to_idx(new_pos, map_size.x);
        idxs[n] = new_idx;

        float new_height = height_map[new_idx];
        float difference = height - new_height;                        // positive if center is higher
        difference = difference > slope_threshold ? difference : 0.0f; // threshold

        if (offset_count > 4 && n > 4) difference /= math::SQRT2; // if we have more than 4 neighbours, the next 8 are diagonal so adjust weight

        slopes[n] = difference;
        total_slope += difference;
    }
    // ================================================================
    // [Distribute]
    // ----------------------------------------------------------------
    if (total_slope <= 0.0f) return; // prevent div by 0
    float erosion = erosion_rate * total_slope;
    if (erosion_yield > 0.0f)
        for (int n = 0; n < offset_count; ++n) {
            if (slopes[n] > 0) {
                float share = (slopes[n] / total_slope) * erosion * erosion_yield;
                int new_idx = idxs[n];
                atomicAdd(&height_map[new_idx], share);
            }
        }
    atomicAdd(&height_map[idx], -erosion);
}

#pragma region UNUSED_R_DIRECTION // unused logic, using normal

// hash_float_signed returns a signed float in [-1,1]
// mode 0: two hashes
// mode 1: ⚠️ EXPERIMETAL, reuse same hash but likely will likely have directional bias
DH_INLINE float2 random_direction_with_magnitude2(const int x, const int y, const int z, const int seed, const int mode = 0) {

    float2 result;
    switch (mode) {
    case 0: {
        float r1 = math::hash_float_signed(x, y, z, seed);        // First hash → angle in [0, 2π)
        float angle = r1 * math::PI;                              // scale [-1,1] → [-π, π]
        float magnitude = math::hash_float(x, y, z, seed + 1);    // Second hash → magnitude in [0,1]
        result = make_float2(cos(angle) * magnitude, sin(angle) * magnitude); // Final vector
        break;
    }
    case 1: {
        float r = math::hash_float(x, y, z, seed);
        float angle = r * math::PI * 2.0; // [0, 2π]
        result = make_float2(cos(angle) * r, sin(angle) * r);
        break;
    }
    }
    return result;
}
#pragma endregion

#pragma region PREASSURE_DISTRO_SCHEME // idea of preassure based distro (hard to use slopes)

// preassure distro scheme
struct Flux8 {
    float values[8]; // individual components
};

constexpr float CARDINAL_WEIGHT = 1.0f;
constexpr float DIAGONAL_WEIGHT = math::INV_SQRT2;

constexpr float WEIGHT_SUM = 4 * CARDINAL_WEIGHT + 4 * DIAGONAL_WEIGHT;

constexpr float CARDINAL_NORM = CARDINAL_WEIGHT / WEIGHT_SUM;
constexpr float DIAGONAL_NORM = DIAGONAL_WEIGHT / WEIGHT_SUM;

Flux8 distribute_pressure(float pressure) {
    Flux8 result{};

    // 4 cardinals
    for (int i = 0; i < 4; ++i)
        result.values[i] = pressure * CARDINAL_NORM;

    // 4 diagonals
    for (int i = 4; i < 8; ++i)
        result.values[i] = pressure * DIAGONAL_NORM;

    return result;
}

#pragma endregion

struct Flux9 {
    float values[8]; // individual components
    float total;     // aggregate
};

constexpr float2 DOT_OFFSETS_8[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {0.5, 0.5}, {-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}};
constexpr float INV_DISTANCE[8] = {1.0f, 1.0f, 1.0f, 1.0f, math::INV_SQRT2, math::INV_SQRT2, math::INV_SQRT2, math::INV_SQRT2};

// translates a vector into
DH_INLINE Flux9 dot_flux_calculation(float2 v, bool positive_only = true) {

    Flux9 result;
    result.total = 0.0f;

    for (int n = 0; n < 8; ++n) {
        float2 unit_offset = DOT_OFFSETS_8[n];                 // cell offset as a float vector, not normalized as this helps scale
        float product = math::dot(unit_offset, v); // dot product gets how strongly we push into this direction
        if (positive_only) product = max(product, 0.0);        // positive only
        result.values[n] = product;
        result.total += product;
    }
    return result;
}

// dust guided by wind
__global__ void run_wind(
    const Parameters *__restrict__ pars,
    const ArrayPtrs *__restrict__ arrays,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = math::pos_to_idx(pos, map_size.x);
    int idx2 = idx * 2;
    // ================================================================

    auto _wind_vector2 = arrays->_wind_vector2;
    float2 current_wind = make_float2(_wind_vector2[idx2], _wind_vector2[idx2 + 1]);

    float2 random_normal = math::normal_vector2_fast(pos.x, pos.y, step, 2834);

    // current_wind = math::lerp(current_wind, random_normal, 0.125);
    current_wind += random_normal * pars->wind_strength;

    // // normalize (all add up to 1.0)
    // if (flux_total > 1e-6f) {
    //     for (int n = 0; n < 8; ++n) { fluxes[n] /= flux_total; }
    // } else {
    //     for (int n = 0; n < 8; ++n) { fluxes[n] = 0.0f; } // prevents div by zero (just set 0)
    // }
}

#pragma endregion

#pragma region MAIN

__global__ void calculate_slope_vectors_kernel(
    const float *__restrict__ height_map1, // heightmap (required)
    const float *__restrict__ height_map2, // or null
    const float *__restrict__ height_map3, // or null
    float *__restrict__ slope_vectors_out, // must be double size of height_maps (interleaved the vectors)
    const int2 map_size,
    const bool wrap = true,    // wrap coordinates
    const float jitter = 0.0f, // if > 0.0 apply jitter
    const int step = 0,        // used by jitter, needs to be a different value each step
    const int jitter_mode = 0, // 0 is economical and less accurate
    const float scale = 1.0f,  // larger scale will make slopes less steep
    const int jitter_seed = 1234) {
    // ================================================================
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = math::pos_to_idx(pos, map_size.x) * 2;
    // ================================================================
    float2 slope_vector = math::compute_slope_vector_OLD(height_map1, height_map2, height_map3, map_size, pos, wrap, jitter, step, jitter_mode, scale, jitter_seed);
    slope_vectors_out[idx] = slope_vector.x;
    slope_vectors_out[idx + 1] = slope_vector.y;
}

void TEMPLATE_CLASS_NAME::allocate_device() {

    auto &pars = _pars.host();

    if (_device_allocated) { return; }

    if (pars._debug) { printf("⚠️  debug mode active!\n"); }

    if (!layer_map.empty()) {
        printf("🐡 layer_map detected...\n");
        core::logging::println("shape: ", height_map.shape());

        pars._width = layer_map.shape()[0];
        pars._height = layer_map.shape()[1];
        pars._layers = layer_map.shape()[2]; // marks layers mode as active

        height_map.set_stream(stream.get());
        height_map.resize({pars._width, pars._height}); // allocate the heightmap still, we will copy the total height to it

        if (pars.sediment_layer_mode) {
            sediment_layer_map.resize(layer_map.shape());
        }

#ifdef PRECALCULATE_EXPOSED_LAYER
        _exposed_layer_map.resize(height_map.shape()); // set _exposed_layer_map to height_map dimensions
#endif

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
        core::logging::println("shape: ", height_map.shape());

        pars._width = height_map.shape()[0];
        pars._height = height_map.shape()[1];
        // pars._layers = 1;
    } else {
        throw std::runtime_error("layer_map and height_map empty!");
    }

    // ================================================================

    // size_t array_size = pars._width * pars._height;

    // _flux8.resize({array_size * 8});                       // flux output
    // _sediment_flux8.resize({array_size * 8});              // sediment flux output

    _flux8.resize({pars._width, pars._height, 8});
    _sediment_flux8.resize({pars._width, pars._height, 8});

    _slope_vector2.resize({pars._width, pars._height, 2}); // 2D vectors

    if (pars.sea_pass) {
        _sea_map.resize(height_map.shape());
    }

    switch (pars.mode) {
    case 1: {                                                 // TEST mode
        _wind_vector2.resize({pars._width, pars._height, 2}); // testing wind
        break;
    }
    }

    // ================================================================

// allocate and zero arrays
#define ZERO_ARRAYS \
    X(water_map)    \
    X(sediment_map)
#define X(NAME)                               \
    if (NAME.empty()) {                       \
        NAME.resize(height_map.shape()); \
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
    NAME.resize(height_map.shape());
    ALLOCATE_ARRAYS
#undef X
#undef ALLOCATE_ARRAYS

    // ================================================================

    dev_array_ptrs.upload(get_array_ptrs()); // must get the pointers AFTER allocation

    _device_allocated = true;
}

void TEMPLATE_CLASS_NAME::process() {
    // printf("process()...\n");

    auto &pars = _pars.host(); // simple alias
    const auto &pars_ptr = _pars.dev_ptr();

    switch (pars.mode) {
    case 1:
        _process1();
        break;
    case 0:
        _process2();
        break;
    default:
        throw std::runtime_error(
            "invalid mode: " + std::to_string(pars.mode));
    }
}

void TEMPLATE_CLASS_NAME::_process1() {
    // printf("_process()...\n");

    allocate_device();
    configure_device();
    stream.sync();

    auto &pars = _pars.host(); // simple alias
    const auto &pars_ptr = _pars.dev_ptr();

    for (int i = 0; i < pars.steps; ++i) {
        // ✔️ layer_mode_calculations3
        if (pars._layers > 0) {
            layer_mode_calculations3<<<grid, block, 0, stream.get()>>>(
                pars_ptr,
                dev_array_ptrs.dev_ptr());
        }
        // ✔️ add_rain3
        if (pars.rain_rate > 0.0f) {
            add_rain3<<<grid, block, 0, stream.get()>>>(
                pars_ptr,
                dev_array_ptrs.dev_ptr());
        }
        // ✔️ main erosion
        if (pars._main_loop) {
            calculate_outflow3<<<grid, block, 0, stream.get()>>>(
                pars_ptr,
                dev_array_ptrs.dev_ptr(),
                debug_outputs.dev_ptr(),
                pars._step);
            apply_flux3<<<grid, block, 0, stream.get()>>>(
                pars_ptr,
                dev_array_ptrs.dev_ptr(),
                debug_outputs.dev_ptr(),
                pars._step);
        }
        // 🚧 sea_pass3
        if (pars.sea_pass) {
            sea_pass3<<<grid, block, 0, stream.get()>>>(
                pars_ptr,
                dev_array_ptrs.dev_ptr(),
                debug_outputs.dev_ptr(),
                pars._step);
        }
        // ✔️ simple_collapse4
        if (pars.simple_collapse) {
            simple_collapse4<<<grid, block, 0, stream.get()>>>(
                pars_ptr,
                height_map.dev_ptr(),
                pars._step);
        }
        pars._step++;
    }

    stream.sync(); // ⚠️ ?? needed
}

void TEMPLATE_CLASS_NAME::_process2() {
    // printf("_process_test()...\n");

    // allocate_device();
    configure_device();
    stream.sync();

    auto &pars = _pars.host(); // simple alias
    const auto &pars_ptr = _pars.dev_ptr();

    for (int i = 0; i < pars.steps; ++i) {

        // STAGE_CALCULATE_SLOPES();

        STAGE_RAIN();
        STAGE_MAIN();

        STAGE_SEA_PASS();

        STAGE_SIMPLE_COLLAPSE();

        // layer_mode_calculations3<<<grid, block, 0, stream.get()>>>(
        //     pars_ptr,
        //     dev_array_ptrs.dev_ptr());

        pars._step++;
    }

    stream.sync(); // ⚠️ ?? needed
}

void TEMPLATE_CLASS_NAME::debug_update() {
    // pars = dev_pars.download();
    // stream.sync();
    // cudaDeviceSynchronize(); // required as we didn't implement stream in dev_pars

    debug_outputs.set_stream(stream.get());
    debug_outputs.download();
}

#pragma endregion

#pragma region STAGES

void TEMPLATE_CLASS_NAME::STAGE_RAIN() {
    auto stage = Stage::RAIN;
    auto &pars = _pars.host();
    const auto &pars_ptr = _pars.dev_ptr();

    if (pars.rain_rate > 0.0f) {

        if (!_stage_configured.count(stage)) {
            allocate_water_map();

            dev_array_ptrs.upload(get_array_ptrs());
            stream.sync();

            _stage_configured.insert(stage);
        }

        add_rain3<<<grid, block, 0, stream.get()>>>(_pars.dev_ptr(), dev_array_ptrs.dev_ptr());
    }
}

void TEMPLATE_CLASS_NAME::STAGE_CALCULATE_SLOPES() {
    auto stage = Stage::CALCULATE_SLOPES;
    auto &pars = _pars.host();
    const auto &pars_ptr = _pars.dev_ptr();

    if (!_stage_configured.count(stage)) {
        allocate__slope_vector2();

        dev_array_ptrs.upload(get_array_ptrs());
        stream.sync();
        _stage_configured.insert(stage);
    }

    const int jitter_seed = 1234;

    calculate_slope_vectors_kernel<<<grid, block, 0, stream.get()>>>(
        height_map.dev_ptr(),
        nullptr,
        nullptr,
        _slope_vector2.dev_ptr(),
        make_int2(_pars.host()._width, _pars.host()._height),
        _pars.host().wrap,
        _pars.host().slope_jitter,
        _pars.host()._step,
        _pars.host().slope_jitter_mode,
        _pars.host().scale,
        jitter_seed);
}

void TEMPLATE_CLASS_NAME::STAGE_SIMPLE_COLLAPSE() {
    auto stage = Stage::SIMPLE_COLLAPSE;
    auto &pars = _pars.host();
    const auto &pars_ptr = _pars.dev_ptr();

    if (!pars.simple_collapse) return;

    if (!_stage_configured.count(stage)) {

        allocate_height_map(); // only uses the error

        dev_array_ptrs.upload(get_array_ptrs());
        stream.sync();
        _stage_configured.insert(stage);
    }

    simple_collapse4<<<grid, block, 0, stream.get()>>>(
        pars_ptr,
        height_map.dev_ptr(),
        pars._step);
}

void TEMPLATE_CLASS_NAME::STAGE_MAIN() {

    auto stage = Stage::MAIN;
    auto &pars = _pars.host();
    const auto &pars_ptr = _pars.dev_ptr();

    if (!pars._main_loop) return;

    if (!_stage_configured.count(stage)) {

        // if the height_map is empty and the layer_map not, we want to use layer mode
        if (height_map.empty() && !layer_map.empty()) {
            auto &dim = layer_map.shape();
            height_map.resize({dim[0], dim[1]});
            pars._layers = dim[2];
        }

        if (height_map.empty()) throw std::runtime_error("height_map is empty");

        allocate_water_map();
        allocate_sediment_map();
        allocate__slope_vector2();   // ❓ UNUSED so far
        allocate__slope_magnitude(); // ❓ USED FOR SOME EROSION MODES
        allocate__water_velocity();  // ❓USED FOR SOME EROSION MODES
        allocate__flux8();
        allocate__sediment_flux8();
        allocate__water_out();    // ❗ avoids recalculating
        allocate__sediment_out(); // ❗ avoids recalculating

        dev_array_ptrs.upload(get_array_ptrs());
        stream.sync();

        _stage_configured.insert(stage);
    }

    if (pars._layers > 0) {
        layer_mode_calculations3<<<grid, block, 0, stream.get()>>>(
            pars_ptr,
            dev_array_ptrs.dev_ptr());
    }

    calculate_outflow3<<<grid, block, 0, stream.get()>>>(
        pars_ptr,
        dev_array_ptrs.dev_ptr(),
        debug_outputs.dev_ptr(),
        pars._step);
    apply_flux3<<<grid, block, 0, stream.get()>>>(
        pars_ptr,
        dev_array_ptrs.dev_ptr(),
        debug_outputs.dev_ptr(),
        pars._step);
}

void TEMPLATE_CLASS_NAME::STAGE_SEA_PASS() {
    auto stage = Stage::SEA_PASS;
    auto &pars = _pars.host();
    const auto &pars_ptr = _pars.dev_ptr();

    if (!pars.sea_pass) return;

    if (!_stage_configured.count(stage)) {
        allocate_height_map();
        allocate_water_map();
        allocate_sediment_map();
        allocate__sea_map();

        dev_array_ptrs.upload(get_array_ptrs());
        stream.sync();
        _stage_configured.insert(stage);
    }

    sea_pass3<<<grid, block, 0, stream.get()>>>(
        pars_ptr,
        dev_array_ptrs.dev_ptr(),
        debug_outputs.dev_ptr(),
        pars._step);
}

#pragma endregion

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
