#include "core/cuda/math.cuh"
#include "core/math.h"
#include "gnc/gnc_erosion.cuh"

namespace TEMPLATE_NAMESPACE {

namespace math = core::math;
namespace cmath = core::cuda::math;

// __device__ __constant__ int2 OFFSETS[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // 8 offsets with the opposites in pairs, first 4 cardinal
// __device__ __constant__ float OFFSET_DISTANCES[8] = {1.0f, 1.0f, 1.0f, 1.0f, SQRT2, SQRT2, SQRT2, SQRT2};
// __device__ __constant__ int OFFSET_OPPOSITE_REFS[8] = {1, 0, 3, 2, 5, 4, 7, 6};
// __device__ __constant__ float2 UNIT_OFFSETS_8[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {INV_SQRT2, INV_SQRT2}, {-INV_SQRT2, -INV_SQRT2}, {INV_SQRT2, -INV_SQRT2}, {-INV_SQRT2, INV_SQRT2}};

// E, W, N, S, SE, NW, NE, SW
static constexpr int2 OFFSETS[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

// get total height of ground iterating the layers
D_INLINE float get_layered_height(
    const float *__restrict__ layer_map,
    const int layers,
    const int layer_idx) {

    float height = 0.0;
    for (int n = 0; n < layers; ++n) {
        height += layer_map[layer_idx + n];
    }
    return height;
}

// return exposed layer id, or if all layers empty return invalid ref == to total layers
D_INLINE int get_exposed_layer(
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

// a kernel example makes a chequer pattern
__global__ void layer_calculations(
    const int width, const int height, const int layer_count,
    const float *__restrict__ layer_map, // in
    float *__restrict__ height_map,      // out
    int *__restrict__ _exposed_layer     // out
) {
    // ================================================================
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    int layer_idx = idx * layer_count;
    // ================================================================
    float total_height = 0.0f;
    int exposed_layer = -1; // top layer (-1 is invalid)

    for (int n = layer_count - 1; n >= 0; --n) {
        float value = layer_map[layer_idx + n];
        total_height += value;

        if (exposed_layer == -1 && value > 0.0f)
            exposed_layer = n; // detected exposed layer
    }

    height_map[idx] = total_height;
    _exposed_layer[idx] = exposed_layer;
}

__global__ void add_rain(
    const int width, const int height,
    float *__restrict__ water_map,
    float *__restrict__ rain_map,
    const float rain_rate) {
    // ================================================================
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    // ================================================================
    float rain = rain_rate;
    if (rain_map) rain *= rain_map[idx];
    water_map[idx] += rain;
}

__global__ void calculate_slope_vectors(
    const int2 map_size,
    const float *__restrict__ height_map, // in
    const float *__restrict__ water_map,  // in
    float *__restrict__ _slope_vector2,   // out
    const float jitter,
    const int step,
    const bool wrap,
    const int jitter_mode,
    const float scale,
    const int jitter_seed) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    int idx = cmath::pos_to_idx(pos, map_size);
    int idx2 = idx * 2;
    // ================================================================

    // float jitter = 1.0f;
    // int step = 0;
    // bool wrap = true;
    // int jitter_mode = 0;
    // float scale = 1.0f;
    // int jitter_seed = 1234;

    float2 slope_vector2 = cmath::calculate_slope_vector(
        height_map, water_map, nullptr,
        map_size, pos, wrap, jitter, step, jitter_mode, scale, jitter_seed);
    _slope_vector2[idx2] = slope_vector2.x;
    _slope_vector2[idx2 + 1] = slope_vector2.y;
}

// // std::array in kernel
// __global__ void foo(std::array<float, 4> arr) {
//     float x = arr[2];
// }

__global__ void calculate_outflow3(
    const Parameters *__restrict__ pars,
    const ArrayPointers *__restrict__ arrays,
    // DebugOutputs *__restrict__ debug,
    const int step) {
    // ================================================================
    // int2 map_size = make_int2(pars->_width, pars->_height);
    // int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    // if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
    //     return;
    // int idx = math::pos_to_idx(pos, map_size.x);

    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    int idx = cmath::pos_to_idx(pos, map_size);
    // int idx2 = idx * 2;

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

    int xp = cmath::wrap_or_clamp_index(pos.x + 1, map_size.x, pars->wrap); // x + 1
    int xn = cmath::wrap_or_clamp_index(pos.x - 1, map_size.x, pars->wrap); // x - 1
    int yp = cmath::wrap_or_clamp_index(pos.y + 1, map_size.y, pars->wrap); // y + 1
    int yn = cmath::wrap_or_clamp_index(pos.y - 1, map_size.y, pars->wrap); // y - 1

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
            uint32_t h = cmath::hash_uint(pos.x, pos.y, step, 0);
            xp_surface += cmath::hash_to_4randf(h, 0) * pars->slope_jitter;
            yp_surface += cmath::hash_to_4randf(h, 1) * pars->slope_jitter;
            xn_surface += cmath::hash_to_4randf(h, 2) * pars->slope_jitter;
            yn_surface += cmath::hash_to_4randf(h, 3) * pars->slope_jitter;
            break;
        }
        case 1: { // uses 4 hashes, technically better random
            xp_surface += cmath::hash_float_signed(pos.x, pos.y, step, 0) * pars->slope_jitter;
            yp_surface += cmath::hash_float_signed(pos.x, pos.y, step, 1) * pars->slope_jitter;
            xn_surface += cmath::hash_float_signed(pos.x, pos.y, step, 2) * pars->slope_jitter;
            yn_surface += cmath::hash_float_signed(pos.x, pos.y, step, 3) * pars->slope_jitter;
            break;
        }
        }
    }
    // ----------------------------------------------------------------
    float2 slope_vector = float2{xn_surface - xp_surface, yn_surface - yp_surface}; // note slope may be double actual (use scale to compensate)
    slope_vector /= pars->scale;                                                    // scale such that double world size would mean half gradients
    // ----------------------------------------------------------------

    int idx2 = idx * 2;
    arrays->_slope_vector2_map[idx2] = slope_vector.x; // save to a vector map for later use
    arrays->_slope_vector2_map[idx2 + 1] = slope_vector.y;

    float slope_magnitude = cmath::length(slope_vector); // 🧪 save magnitude (OPTIONAL)
    arrays->_slope_magnitude_map[idx] = slope_magnitude;

    float water_velocity = pow(water, 2.0f / 3.0f) * sqrt(slope_magnitude); // 🧪 manning based velocity??
    arrays->_water_velocity_map[idx] = water_velocity;

    // ================================================================
    // Calculate Fluxes
    // ----------------------------------------------------------------

    // these offsets scale the result of the diagonals down by using a magnitude of 1/SQRT(2), the axis of this is 0.5
    constexpr float2 DOT_OFFSETS_8[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {0.5, 0.5}, {-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}};

    float fluxes[8];
    float flux_total = 0.0f;
    for (int n = 0; n < 8; ++n) {
        float2 unit_offset = DOT_OFFSETS_8[n];                 // cell offset as a float vector, not normalized as this helps scale
        float product = cmath::dot(unit_offset, slope_vector); // dot product gets how strongly we push into this direction
        product = max(product, 0.0);                           // positive only
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
    float *_flux8_ptr = &arrays->_flux8_map[idx8];
    float *_sediment_flux8_ptr = &arrays->_sediment_flux8_map[idx8];
    for (int n = 0; n < 8; ++n) {
        float flux = fluxes[n];
        _flux8_ptr[n] = flux * water_outflow;
        _sediment_flux8_ptr[n] = flux * sediment_outflow;
    }

    // save the total outflow's rather than add them up again later
    arrays->_water_out[idx] = water_outflow;
    arrays->_sediment_out_map[idx] = sediment_outflow;
}

void TEMPLATE_CLASS_NAME::_compute() {

    if (layer_map.is_valid() && !layer_map->empty()) { // layer mode
        _layer_mode = true;
        height_map.instantiate_if_null();
        auto shape = layer_map->shape();
        height_map->resize(shape[0], shape[1]);
        _layer_count = shape[2];

    } else if (height_map.is_valid() && !height_map->empty()) { // heightmap only
        _layer_mode = false;
        _layer_count = 1;

    } else {
        throw std::runtime_error("layermap or heightmap is not valid");
    }

    auto height_map_shape = height_map->shape();
    _width = height_map->width();
    _height = height_map->height();

    _exposed_layer_map.instantiate_if_null();
    _exposed_layer_map->resize(_width, _height);

    _slope_vector2_map.instantiate_if_null();
    _slope_vector2_map->resize(_width, _height, 2);

    water_map.instantiate_if_null();
    if (water_map->shape() != height_map_shape) {
        water_map->resize(height_map_shape);
        water_map->zero_device();
    }

    stream.instantiate_if_null();

    dim3 block(16, 16);
    dim3 grid((_width + block.x - 1) / block.x, (_height + block.y - 1) / block.y);

    int2 map_size = {_width, _height};

    // calculate the layer height for layer mode
    if (_layer_mode) {
        layer_calculations<<<grid, block, 0, stream->get()>>>(
            _width, _height, _layer_count,
            layer_map->dev_ptr(),         // in
            height_map->dev_ptr(),        // out
            _exposed_layer_map->dev_ptr() // out
        );
    }

    calculate_slope_vectors<<<grid, block, 0, stream->get()>>>(
        map_size,
        height_map->dev_ptr(),         // in
        water_map->dev_ptr(),          // in
        _slope_vector2_map->dev_ptr(), // out
        slope_jitter,
        _step,
        true,
        0,    // jitter mode
        1.0f, // scale
        CONSTEXPR_LINE_SEED);

    _step++;

    // output.instantiate_if_null();           // if no DeviceArray make one
    // *output.shared_ptr = *input.shared_ptr; // will copy the memory (on the gpu) from input to output (by dereferencing)
}

} // namespace TEMPLATE_NAMESPACE
