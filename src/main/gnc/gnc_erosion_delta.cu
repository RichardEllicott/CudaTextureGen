#include "gnc/gnc_erosion_delta.cuh"

#include "core/cuda/grid.cuh"
#include "core/cuda/math/grid.cuh"

namespace TEMPLATE_NAMESPACE {

// ================================================================================================================================

void TEMPLATE_CLASS_NAME::GPU_calculate_layer() {
    if (_layer_count > 0) {
        cmath::grid::layer_info_kernel<<<_grid, _block, 0, stream->get()>>>(
            _size,
            _layer_count,

            layer_map->dev_ptr(),         // in
            height_map->dev_ptr(),        // out
            _exposed_layer_map->dev_ptr() // out
        );
    }
}

// ================================================================================================================================
void TEMPLATE_CLASS_NAME::GPU_calculate_slope_vectors() {

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

        1.0f);
}
// ================================================================================================================================
struct DotFluxCalculation {
    float values[8];

    DH_INLINE DotFluxCalculation(float2 vector) {

        constexpr float2 DOT_OFFSETS_8[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {0.5, 0.5}, {-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}};

        float total = 0.0f;

        for (int n = 0; n < 8; ++n) {
            float product = cmath::dot(DOT_OFFSETS_8[n], vector); // dot product gets how strongly we push into this direction
            product = max(product, 0.0f);                         // positive only

            values[n] = product;
            total += product;
        }

        // normalize stage
        if (total > 1e-6f) {
            for (int n = 0; n < 8; ++n) { values[n] /= total; }
        } else {
            for (int n = 0; n < 8; ++n) { values[n] = 0.0f; } // prevents div by zero (just set 0)
        }
    }
};
// --------------------------------------------------------------------------------------------------------------------------------
DH_INLINE float get_water_velocity(int mode, float water, float slope_magnitude) {

    switch (mode) {
    case 0: // manning based
        return cmath::fast::pow(water, 2.0f / 3.0f) * cmath::fast::sqrt(slope_magnitude);
        break;
    case 1: // Chezy, friction?
        return cmath::fast::sqrt(water * slope_magnitude);
        break;
    case 2: // Linear slope model (ultra‑cheap)
        return water;
        break;
    case 3: // Depth‑scaled linear model
        return water * slope_magnitude;
        break;
    case 4: // Square‑root slope only
        return cmath::fast::sqrt(slope_magnitude);
        break;
    case 5: // Depth‑root model (A simple alternative to Chezy)
        return cmath::fast::sqrt(water) * cmath::fast::sqrt(slope_magnitude);
        break;
    case 6: // Power‑law model (generalised Manning/Chezy)
            // return cmath::fast::pow(water, a) * cmath::fast::pow(slope_magnitude, b);
        break;
    case 7: // Clamped velocity model
        // return cmath::fast::pow(water, a) * cmath::fast::pow(slope_magnitude, b);
        break;
    }

    return 0.0f;
}
// --------------------------------------------------------------------------------------------------------------------------------
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
    int idx8 = idx * 8;
    // ----------------------------------------------------------------
    // input maps
    float *height_map = arrays->height_map;                 // in
    float *water_map = arrays->water_map;                   // in
    float *sediment_map = arrays->sediment_map;             // in
    float *_slope_vector2_map = arrays->_slope_vector2_map; // in
    // ----------------------------------------------------------------
    // output maps
    float *_flux8_map = arrays->_flux8_map;                   // out
    float *_sediment_flux8_map = arrays->_sediment_flux8_map; // out
    float *_water_out_map = arrays->_water_out_map;           // out
    float *_sediment_out_map = arrays->_sediment_out_map;     // out
    float *_water_velocity_map = arrays->_water_velocity_map; // out
    // ----------------------------------------------------------------
    // input vars
    float water = water_map[idx];
    float sediment = sediment_map[idx];
    float2 slope_vector = carray::load_float2(_slope_vector2_map, idx2);
    // ----------------------------------------------------------------
    // pars
    int water_velocity_mode = pars->water_velocity_mode;
    float sediment_capacity = pars->sediment_capacity;
    float max_water_outflow = pars->max_water_outflow;
    float flow_rate = pars->flow_rate;
    // ----------------------------------------------------------------

    // calc slope magnitude
    float slope_magnitude = cmath::length(slope_vector);

    // dot flux calculations
    DotFluxCalculation dot_flux(slope_vector);

    // water velocity
    float water_velocity = get_water_velocity(water_velocity_mode, water, slope_magnitude);
    water_velocity *= flow_rate;

    // water outflow
    float water_outflow = water_velocity * water;
    water_outflow = cmath::min(water_outflow, water);             // can't flow more than we have
    water_outflow = cmath::min(water_outflow, max_water_outflow); // capped max flow

    // sediment outflow
    float sediment_outflow = water_outflow * sediment_capacity;
    sediment_outflow = cmath::min(sediment_outflow, sediment); // can't be more than what we have

    // save flux8
    float *_flux8_ptr = &arrays->_flux8_map[idx8];
    float *_sediment_flux8_ptr = &arrays->_sediment_flux8_map[idx8];
    for (int n = 0; n < 8; ++n) {
        float flux = dot_flux.values[n];
        _flux8_ptr[n] = flux * water_outflow;             // save flux out to cell
        _sediment_flux8_ptr[n] = flux * sediment_outflow; // and sediment
    }

    _water_velocity_map[idx] = water_velocity; // save water velocity
    _water_out_map[idx] = water_outflow;       // save water out
    _sediment_out_map[idx] = sediment_outflow;
}
// --------------------------------------------------------------------------------------------------------------------------------
void TEMPLATE_CLASS_NAME::GPU_calculate_flux4() {

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
    int2 map_size = pars->_size;
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    // ================================================================
    int idx = cmath::pos_to_idx(pos, map_size.x);
    int idx2 = idx * 2;
    int idx8 = idx * 8;
    // ----------------------------------------------------------------
    // input maps
    float *height_map = arrays->height_map;                 // in
    float *water_map = arrays->water_map;                   // in
    float *sediment_map = arrays->sediment_map;             // in
    float *_slope_vector2_map = arrays->_slope_vector2_map; // in
    float *rain_map = arrays->rain_map;                     // in
    // ----------------------------------------------------------------
    // output maps
    float *_flux8_map = arrays->_flux8_map;
    float *_sediment_flux8_map = arrays->_sediment_flux8_map;
    float *_water_out_map = arrays->_water_out_map;
    float *_sediment_out_map = arrays->_sediment_out_map;
    float *_water_velocity_map = arrays->_water_velocity_map;

    // ----------------------------------------------------------------
    // input vars
    float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];

    float water_out = arrays->_water_out_map[idx];
    float sediment_out = arrays->_sediment_out_map[idx];
    float water_velocity = _water_velocity_map[idx];

    float rain_map_value = 1.0f;
    if (rain_map) rain_map_value = rain_map[idx];

    // pars
    bool wrap = pars->wrap;
    float min_height = pars->min_height;
    float max_height = pars->max_height;
    int erosion_mode = pars->erosion_mode;
    float erosion_rate = pars->erosion_rate;
    float sediment_yield = pars->sediment_yield;
    float drain_rate = pars->drain_rate;
    float evaporation_rate = pars->evaporation_rate;
    float rain_rate = pars->rain_rate;
    float deposition_rate = pars->deposition_rate;
    int _layer_count = pars->_layer_count;

    // ================================================================
    // [Layers]
    // ----------------------------------------------------------------
    int exposed_layer;
    float layer_erosiveness = 1.0f;
    float layer_sediment_yield = 1.0f;
    if (_layer_count > 0) {
        exposed_layer = arrays->_exposed_layer_map[idx];
        layer_erosiveness = pars->layer_erosiveness_array[exposed_layer];
        layer_sediment_yield = pars->layer_sediment_yield_array[exposed_layer];
    }

    // ================================================================
    // [Flux]
    // ----------------------------------------------------------------

    // float sediment_change = -sediment_out;
    sediment -= sediment_out;
    water -= water_out;

    float *_flux8_ptr = &_flux8_map[idx8];
    float *_sediment_flux8_ptr = &_sediment_flux8_map[idx8];
    for (int n = 0; n < 8; ++n) {
        int2 new_pos = cmath::wrap_or_clamp_index(pos + cmath::constants::GRID_OFFSETS_8[n], map_size, wrap);
        int new_idx = cmath::pos_to_idx(new_pos, map_size);
        int new_idx8 = new_idx * 8;

        int opposite_n = n ^ 1;          // flip bit 1, giving opposite
        water += _flux8_ptr[opposite_n]; //  inflow from neighbouring tiles
        sediment += _sediment_flux8_ptr[opposite_n];
    }

    // ================================================================
    // [Erosion]
    // ----------------------------------------------------------------
    float available_erosion = height - min_height; // limit erosion to available rock above min_height
    float erosion;
    switch (erosion_mode) {
    case 0:
        erosion = water_out; // just outflow
        break;
    case 1:
        erosion = water_out * water_velocity; // outflow * velocity (note water out was already affected by velocity)
    }

    // 🧪 apply other effects before final adjustment

    erosion *= erosion_rate * layer_erosiveness; // multiple by erosion rate and layer erosion rate

    erosion = cmath::clamp(erosion, 0.0f, available_erosion); // ensure not negative and not more than available_erosion
    height -= erosion;                                        // apply erosion to height (not layer mode)
    sediment += erosion * sediment_yield * layer_sediment_yield;
    // ================================================================
    // [Drain]
    // ----------------------------------------------------------------
    if (drain_rate > 0.0f && height <= min_height) {
        water -= drain_rate; // minus drain (could be negative)
    }
    // ================================================================
    // [Evaporate]
    // ----------------------------------------------------------------
    water -= evaporation_rate;
    // ================================================================
    // [Rain]
    // ----------------------------------------------------------------
    water += rain_rate * rain_map_value;
    // ================================================================
    // [Deposition]
    // ----------------------------------------------------------------
    // water -= evaporation_rate;
    float available_sediment = sediment;
    float deposition = cmath::min(deposition_rate, available_sediment);
    height += deposition;
    sediment -= deposition;
    // ================================================================
    // [Output]
    // ----------------------------------------------------------------
    if (_layer_count > 0) {
    }

    height_map[idx] = cmath::clamp(height, min_height, max_height);
    water_map[idx] = cmath::max(water, 0.0f);
    sediment_map[idx] = cmath::max(sediment, 0.0f);
}
// --------------------------------------------------------------------------------------------------------------------------------
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

#define LOG_HERE() printf("DEBUG LOG %s:%d (%s)\n", __FILE__, __LINE__, __func__)
#define LOG_HERE()

void TEMPLATE_CLASS_NAME::setup() {

    if (layer_map.is_valid() && !layer_map->empty()) {

        auto layer_map_shape = layer_map->shape();
        _layer_count = layer_map_shape[2];
        height_map.instantiate_if_null();
        height_map->resize(layer_map_shape[0], layer_map_shape[1]);
    }

    LOG_HERE();

    auto shape = height_map->shape();
    auto shape2 = std::array{shape[0], shape[1], (size_t)2};
    auto shape8 = std::array{shape[0], shape[1], (size_t)8};

    LOG_HERE();
    _size = to_int2(shape);
    _grid = cmath::calculate_grid(_size, _block);

    LOG_HERE();
    ensure_array_ref_ready(water_map, shape, true);
    ensure_array_ref_ready(sediment_map, shape, true);

    LOG_HERE();
    ensure_array_ref_ready(_water_velocity_map, shape, true); // if velocity mode 1
    ensure_array_ref_ready(_water_out_map, shape);
    ensure_array_ref_ready(_sediment_out_map, shape);

    LOG_HERE();
    ensure_array_ref_ready(_slope_vector2_map, shape2);

    LOG_HERE();
    ensure_array_ref_ready(_flux8_map, shape8);
    ensure_array_ref_ready(_sediment_flux8_map, shape8);

    rain_map.instantiate_if_null(); // optional map

    LOG_HERE();
    ready_device();
}

void TEMPLATE_CLASS_NAME::_compute() {

    LOG_HERE();

    setup();

    for (int i = 0; i < steps; i++) {

        GPU_calculate_layer();
        LOG_HERE();

        GPU_calculate_slope_vectors();
        LOG_HERE();

        GPU_calculate_flux4();
        LOG_HERE();

        GPU_apply_flux4();
        LOG_HERE();

        _step++;
    }
}

} // namespace TEMPLATE_NAMESPACE
