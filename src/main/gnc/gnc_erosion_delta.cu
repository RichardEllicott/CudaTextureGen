#include "gnc/gnc_erosion_delta.cuh"

#include "core/cuda/grid.cuh"
#include "core/cuda/math/grid.cuh"

namespace TEMPLATE_NAMESPACE {

// ================================================================================================================================

void TEMPLATE_CLASS_NAME::GPU_calculate_slope_vectors() {

    auto shape = height_map->shape();
    auto shape2 = std::array{shape[0], shape[1], (size_t)2};
    // auto shape8 = std::array{shape[0], shape[1], (size_t)8};

    ensure_array_ref_ready(water_map, shape, true);
    ensure_array_ref_ready(_slope_vector2_map, shape2, true);
    ensure_array_ref_ready(_slope_magnitude_map, shape, true);

    cmath::grid::slope_vector_kernel<<<_grid, _block, 0, stream->get()>>>(
        _size,

        height_map->dev_ptr(), // in
        water_map->dev_ptr(),  // in
        nullptr,
        _slope_vector2_map->dev_ptr(), // out

        wrap,

        0, // jitter mode
        slope_jitter,
        _step,
        0x865C34F3u,

        1.0f,
        _slope_magnitude_map->dev_ptr());
}

// ================================================================================================================================

// struct Flux9 {
//     float values[8];  // individual components
//     float total = {}; // aggregate
// };

// DH_INLINE Flux9 dot_flux_calculation(float2 v, bool positive_only = true) {

//     Flux9 result;
//     // result.total = 0.0f;

//     for (int n = 0; n < 8; ++n) {

//         float2 unit_offset = cmath::GRID_OFFSETS_8_DOTS[n]; // cell offset as a float vector, not normalized as this helps scale

//         float product = cmath::dot(unit_offset, v);             // dot product gets how strongly we push into this direction
//         product = positive_only ? max(product, 0.0f) : product; // positive only

//         result.values[n] = product;
//         result.total += product;
//     }
//     return result;
// }

struct DotFluxCalculation {
    float values[8];
    float total = {};

    static constexpr float2 DOT_OFFSETS_8[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {0.5, 0.5}, {-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}};

    DH_INLINE DotFluxCalculation(float2 vector, bool positive_only = true) {

        for (int n = 0; n < 8; ++n) {
            float product = cmath::dot(DOT_OFFSETS_8[n], vector);   // dot product gets how strongly we push into this direction
            product = positive_only ? max(product, 0.0f) : product; // positive only

            values[n] = product;
            total += product;
        }
    }

    // make all values add up to 1.0f
    DH_INLINE void normalize() {

        if (total > 1e-6f) {
            for (int n = 0; n < 8; ++n) { values[n] /= total; }
        } else {
            for (int n = 0; n < 8; ++n) { values[n] = 0.0f; } // prevents div by zero (just set 0)
        }
    }
};

__global__ void calculate_flux4(
    const Parameters *__restrict__ pars,
    const ArrayPointers *__restrict__ arrays,
    const int step) {
    // ================================================================
    int2 map_size = pars->_size;
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    // ================================================================
    int idx = cmath::pos_to_idx(pos, map_size.x);
    int idx2 = idx * 2;
    // ----------------------------------------------------------------
    float *height_map = arrays->height_map;
    float *water_map = arrays->water_map;
    float *sediment_map = arrays->sediment_map;
    // ================================================================

    float2 slope_vector = carray::load_float2(arrays->_slope_vector2_map, idx2);
    DotFluxCalculation dot_flux(slope_vector); // dot flux calculations
    dot_flux.normalize();




    
}

// --------------------------------------------------------------------------------------------------------------------------------

void TEMPLATE_CLASS_NAME::GPU_calculate_flux4() {

    auto shape = height_map->shape();

    ensure_array_ref_ready(water_map, shape, true);
    ensure_array_ref_ready(sediment_map, shape, true);

    calculate_flux4<<<_grid, _block, 0, stream->get()>>>(
        _dev_pars.dev_ptr(),
        _dev_arrays.dev_ptr(),
        _step);
}

// ================================================================================================================================

__global__ void apply_flux3(
    const Parameters *__restrict__ pars,
    const ArrayPointers *__restrict__ arrays,
    const int step) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    int2 map_size = pars->_size;
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    // ================================================================
    int idx = cmath::pos_to_idx(pos, map_size.x);
    // int idx8 = idx * 8;
    // ----------------------------------------------------------------
    float *height_map = arrays->height_map;
    float *water_map = arrays->water_map;
    float *sediment_map = arrays->sediment_map;
    // ----------------------------------------------------------------
    float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];
    // ================================================================
}

void TEMPLATE_CLASS_NAME::GPU_apply_flux4() {

    apply_flux3<<<_grid, _block, 0, stream->get()>>>(
        _dev_pars.dev_ptr(),
        _dev_arrays.dev_ptr(),
        _step);
}

// ================================================================================================================================

void TEMPLATE_CLASS_NAME::test() {

    printf("test123()...\n");

    auto sfg = core::cuda::grid::SamplingFieldGenerator();
    sfg.print_test_data();
}

void TEMPLATE_CLASS_NAME::_compute() {

    if (!height_map.is_valid()) throw std::runtime_error("height_map is not valid");
    if (height_map->empty()) throw std::runtime_error("height_map is empty");

    _size = to_int2(height_map->shape());

    _grid = cmath::calculate_grid(_size, _block);

    for (int i = 0; i < steps; i++) {

        GPU_calculate_slope_vectors();
        GPU_calculate_flux4();
    }
}

} // namespace TEMPLATE_NAMESPACE
