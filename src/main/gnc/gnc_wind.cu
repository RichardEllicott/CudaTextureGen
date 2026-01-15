#include "core/cuda/math.cuh"
#include "core/cuda/math/array.cuh"
#include "core/cuda/math/grid.cuh"
#include "core/math.h"
#include "gnc/gnc_wind.cuh"

namespace TEMPLATE_NAMESPACE {

namespace math = core::math;
namespace cmath = core::cuda::math;

// ================================================================================================================================

struct Flux9 {
    float values[8];  // individual components
    float total = {}; // aggregate
};

// translates a vector into flux values
D_INLINE Flux9 dot_flux_calculation(float2 v, bool positive_only = true) {

    Flux9 result;
    // result.total = 0.0f;

    for (int n = 0; n < 8; ++n) {

        float2 unit_offset = cmath::GRID_OFFSETS_8_DOTS[n]; // cell offset as a float vector, not normalized as this helps scale

        float product = cmath::dot(unit_offset, v);             // dot product gets how strongly we push into this direction
        product = positive_only ? max(product, 0.0f) : product; // positive only

        result.values[n] = product;
        result.total += product;
    }
    return result;
}

// ================================================================================================================================

DH_INLINE float2 load_float2(const float *base, size_t idx) {
    return cmath::array::load_float2(base, idx);
}

DH_INLINE void store_float2(float *base, size_t idx, float2 v) {
    cmath::array::store_float2(base, idx, v);
}

// simple wind model with slope influence
__global__ void run_wind(
    const Parameters *__restrict__ pars,
    // const ArrayPointers *__restrict__ arrays,

    const int2 map_size,

    const float *__restrict__ wind_vec2_map,  // in
    const float *__restrict__ slope_vec2_map, // in
    float *__restrict__ wind_vec2_map_out,    // out

    const int step) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    int idx = cmath::pos_to_idx(pos, map_size);
    int idx2 = idx * 2;
    // ================================================================
    // [Pars]
    // ----------------------------------------------------------------
    auto random_wind = pars->random_wind;
    auto wind_influence = pars->wind_influence;
    auto slope_influence = pars->slope_influence;
    auto wind_drag = pars->wind_drag;
    float delta_time = 1.0f;
    // ================================================================
    // auto _wind_vector2_map = arrays->wind_vec2;   // wind map
    // auto slope_vec2_map = arrays->slope_vec2_map; // slope map
    float2 slope = load_float2(slope_vec2_map, idx2);
    float2 wind = load_float2(wind_vec2_map, idx2);
    // ================================================================

    wind += cmath::normal_vector2_fast(pos.x, pos.y, step, 0x3A8FB10Au) * random_wind; // random turbulence
    // ================================================================

    for (int n = 0; n < 8; n++) {
        int2 new_pos = pos + cmath::GRID_OFFSETS_8[n];
        new_pos = cmath::posmod(new_pos, map_size);
        int new_idx = cmath::pos_to_idx(new_pos, map_size);
        int new_idx2 = new_idx * 2;
        // ----------------------------------------------------------------
        float2 new_wind = load_float2(wind_vec2_map, new_idx2);                // wind in neighbour tile
        float dot_wind = -cmath::dot(new_wind, cmath::GRID_OFFSETS_8_DOTS[n]); // give a wind dot product scaled so diagonals are penalized
        wind += new_wind * dot_wind * wind_influence;
    }
    // ================================================================
    // [Wind Drag]
    // ----------------------------------------------------------------

    // wind *= 0.999; // optional damp wind?
    wind *= exp(-wind_drag * delta_time);

    // ================================================================
    // [Slope affect]
    // ----------------------------------------------------------------

    float slope_gradient = cmath::length(slope); // gradient can range from 0 (flat) to ∞ (90°)

    if (slope_gradient > 0.0001f) {
        float2 downhill = cmath::normalize(slope);
        float2 wall_normal = downhill * -1.0f;

        float angle = atan(slope_gradient); // angle in radians, [0, ~pi/2)
        const float HALF_PI = 1.57079632679f;
        float slope_strength = angle / HALF_PI; // 0–1

        // --- optional bias (uncomment to test) ---
        // slope_strength = powf(slope_strength, pars->slope_bias);
        // -----------------------------------------

        float into_wall = cmath::dot(wind, wall_normal); // how strongly into the wall

        if (into_wall > 0.0f) {
            wind -= wall_normal * into_wall * slope_strength * slope_influence; // AI gave me this, i think it works
        }
    }

    // ================================================================

    store_float2(wind_vec2_map_out, idx2, wind);
}

// ================================================================================================================================
// __global__ void flux_pass(
//     const Parameters *__restrict__ pars,
//     const ArrayPointers *__restrict__ arrays,
//     const int step) {
// }

// __global__ void apply_pass(
//     const Parameters *__restrict__ pars,
//     const ArrayPointers *__restrict__ arrays,
//     const int step) {
// }
// ================================================================================================================================

// navier stokes vs manning
//
//
// https://copilot.microsoft.com/chats/UzviJqCo12CACrgzFqkVt
//
//
//

void TEMPLATE_CLASS_NAME::_compute() {

    if (height_map.is_null() || height_map->empty())
        throw std::runtime_error("height_map is null or empty");

    auto height_map_shape = height_map->shape();

    size_t _width = height_map_shape[0];
    size_t _height = height_map_shape[1];
    _map_size = {(int)_width, (int)_height};

    // _width = height_map->width();
    // _height = height_map->height();

    ensure_array_ref_ready(slope_vec2_map, std::array<size_t, 3>{_width, _height, 2});
    ensure_array_ref_ready(wind_vec2_map, std::array<size_t, 3>{_width, _height, 2}, true);
    ensure_array_ref_ready(wind_vec2_map_out, std::array<size_t, 3>{_width, _height, 2});
    ensure_array_ref_ready(dust_map, height_map_shape, true);

    dim3 block(16, 16);
    dim3 grid((_width + block.x - 1) / block.x, (_height + block.y - 1) / block.y);

    ready_device();

    // precalculate slope vectors
    cmath::grid::slope_vector_kernel<<<grid, block, 0, stream->get()>>>(
        _map_size,

        height_map->dev_ptr(), // in
        nullptr,
        nullptr,
        slope_vec2_map->dev_ptr(), // out

        wrap);

    // wind update
    run_wind<<<grid, block, 0, stream->get()>>>(
        _dev_pars.dev_ptr(),
        // dev_array_pointers.dev_ptr(),

        _map_size,
        wind_vec2_map->dev_ptr(),     // in
        slope_vec2_map->dev_ptr(),    // in
        wind_vec2_map_out->dev_ptr(), // out

        _step);

    // std::swap(wind_vec2_map.get(), wind_vec2_map_out.get()); // could use the swap built into my arrays
    std::swap(wind_vec2_map, wind_vec2_map_out); // but using the Ref
}

} // namespace TEMPLATE_NAMESPACE
