

#include "gnc/gnc_erosion.cuh"

#include "core/cuda/math/array.cuh"
#include "core/cuda/math/grid.cuh"

#define PRECALCULATE_EXPOSED_LAYER

namespace TEMPLATE_NAMESPACE {

// ================================================================================================================================
#pragma region HELPERS

#pragma endregion
// ================================================================================================================================
#pragma region KERNELS

__global__ void add_rain3(
    const int2 map_size,
    float *__restrict__ water_map, // out
    const float rain_rate,

    const float *__restrict__ rain_map, // in (optional)

    float rain_probability = 1.0f, // [0, 1] leave 1 to disable
    int step = 0) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    int idx = cmath::pos_to_idx(pos, map_size);
    // ================================================================
    // [Rain]
    // ----------------------------------------------------------------
    float rain = rain_rate;
    if (rain_map) {
        rain *= rain_map[idx]; // multiply by rain_map if != nullptr
    }
    water_map[idx] += rain;
}

// --------------------------------------------------------------------------------------------------------------------------------

__global__ void calculate_flux3(
    const Parameters *__restrict__ pars,
    const ArrayPointers *__restrict__ arrays,
    const int step) {
    // ================================================================================================================================
    int2 map_size = pars->_size;
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = cmath::pos_to_idx(pos, map_size.x);
    int idx2 = idx * 2;
    // ================================================================================================================================
    float *height_map = arrays->height_map;
    float *water_map = arrays->water_map;
    float *sediment_map = arrays->sediment_map;
    // ================================================================================================================================
    // float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];
    // float surface = height + water;

    float2 slope_vector = cmath::array::load_float2(arrays->_slope_vector2_map, idx2);

    // ================================================================================================================================
    // Calculate Slope Vector
    // --------------------------------------------------------------------------------------------------------------------------------

    // int xp = cmath::wrap_or_clamp_index(pos.x + 1, map_size.x, pars->wrap); // x + 1
    // int xn = cmath::wrap_or_clamp_index(pos.x - 1, map_size.x, pars->wrap); // x - 1
    // int yp = cmath::wrap_or_clamp_index(pos.y + 1, map_size.y, pars->wrap); // y + 1
    // int yn = cmath::wrap_or_clamp_index(pos.y - 1, map_size.y, pars->wrap); // y - 1

    // int xp_idx = pos.y * map_size.x + xp; // {+1,0}
    // int xn_idx = pos.y * map_size.x + xn; // {-1,0}
    // int yp_idx = yp * map_size.x + pos.x; // {0,+1}
    // int yn_idx = yn * map_size.x + pos.x; // {0,-1}

    // // positive offsets data
    // float xp_height = height_map[xp_idx];
    // float yp_height = height_map[yp_idx];
    // float xp_water = water_map[xp_idx];
    // float yp_water = water_map[yp_idx];
    // float xp_surface = xp_height + xp_water;
    // float yp_surface = yp_height + yp_water;

    // // negative offsets data
    // float xn_height = height_map[xn_idx];
    // float yn_height = height_map[yn_idx];
    // float xn_water = water_map[xn_idx];
    // float yn_water = water_map[yn_idx];
    // float xn_surface = xn_height + xn_water;
    // float yn_surface = yn_height + yn_water;
    // // --------------------------------------------------------------------------------------------------------------------------------
    // // optional jitter
    // if (pars->slope_jitter) {
    //     switch (pars->slope_jitter_mode) {
    //     case 0: { // cheaper, reuses one hash, lower quality random shouldn't be a problem over frames
    //         uint32_t h = cmath::hash_uint(pos.x, pos.y, step, 0);
    //         xp_surface += cmath::hash_to_4randf(h, 0) * pars->slope_jitter;
    //         yp_surface += cmath::hash_to_4randf(h, 1) * pars->slope_jitter;
    //         xn_surface += cmath::hash_to_4randf(h, 2) * pars->slope_jitter;
    //         yn_surface += cmath::hash_to_4randf(h, 3) * pars->slope_jitter;
    //         break;
    //     }
    //     case 1: { // uses 4 hashes, technically better random
    //         xp_surface += cmath::hash_float_signed(pos.x, pos.y, step, 0) * pars->slope_jitter;
    //         yp_surface += cmath::hash_float_signed(pos.x, pos.y, step, 1) * pars->slope_jitter;
    //         xn_surface += cmath::hash_float_signed(pos.x, pos.y, step, 2) * pars->slope_jitter;
    //         yn_surface += cmath::hash_float_signed(pos.x, pos.y, step, 3) * pars->slope_jitter;
    //         break;
    //     }
    //     }
    // }
    // // --------------------------------------------------------------------------------------------------------------------------------
    // float2 slope_vector = float2{xn_surface - xp_surface, yn_surface - yp_surface}; // note slope may be double actual (use scale to compensate)
    // slope_vector /= pars->scale;                                                    // scale such that double world size would mean half gradients
    // ================================================================================================================================

    arrays->_slope_vector2_map[idx2] = slope_vector.x; // save to a vector map for later use
    arrays->_slope_vector2_map[idx2 + 1] = slope_vector.y;

    float slope_magnitude = cmath::length(slope_vector); // 🧪 save magnitude (OPTIONAL)
    // arrays->_slope_magnitude[idx] = slope_magnitude; // 🧪 🧪 🧪 🧪  ? save here? fast to compute

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
    arrays->_water_out_map[idx] = water_outflow;
    arrays->_sediment_out_map[idx] = sediment_outflow;
}

// --------------------------------------------------------------------------------------------------------------------------------

__global__ void apply_flux3(
    const Parameters *__restrict__ pars,
    const ArrayPointers *__restrict__ arrays,
    const int step) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    int2 map_size = pars->_size;
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    int idx = cmath::pos_to_idx(pos, map_size.x);
    // int idx8 = idx * 8;
    // ================================================================
    float *height_map = arrays->height_map;
    float *water_map = arrays->water_map;
    float *sediment_map = arrays->sediment_map;
    // ================================================================
    float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];

    // auto layer_erosiveness_array = pars->layer_erosiveness_array.data(); // array ptr ⚠️ we need a new container!

    // float surface = height + water;
    // ================================================================
    // Find Exposed Layer (if layer mode active)
    // ----------------------------------------------------------------
    int exposed_layer;
    int layer_idx;
    if (pars->_layer_mode_enabled) {
        layer_idx = idx * pars->_layer_count;

#ifdef PRECALCULATE_EXPOSED_LAYER
        exposed_layer = arrays->_exposed_layer_map[idx];
#else
        exposed_layer = get_exposed_layer(arrays->layer_map, pars->_layers, layer_idx);
#endif
    }
    // ================================================================
    // [Flow]
    // ----------------------------------------------------------------
    float water_out = arrays->_water_out_map[idx];           // water out already calculated
    float sediment_change = -arrays->_sediment_out_map[idx]; // sediment out already calculated
    float water_in = 0.0f;                                   // to calculate from neighbours

    for (int n = 0; n < 8; ++n) {
        int2 new_pos = cmath::wrap_or_clamp_index(pos + cmath::GRID_OFFSETS_8[n], map_size, pars->wrap);
        int new_idx = cmath::pos_to_idx(new_pos, map_size.x);
        int new_idx8 = new_idx * 8;
        int opposite_ref = cmath::GRID_OFFSETS_8_OPPOSITE_INDEX[n];

        water_in += arrays->_flux8_map[new_idx8 + opposite_ref]; //  inflow from neighbouring tiles
        sediment_change += arrays->_sediment_flux8_map[new_idx8 + opposite_ref];
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
        erosion = water_out * pars->erosion_rate * arrays->_slope_magnitude_map[idx]; // with slope_magnitude ⚠️ BROKE?
        break;
    case 2:
        erosion = water * pars->erosion_rate * arrays->_slope_magnitude_map[idx]; // maybe based on total water?
        break;
    case 3:
        erosion = water * pars->erosion_rate * arrays->_water_velocity_map[idx]; // total water and the water velocity (manning)
        break;
    case 4: // soft saturation scheme (limits the max erosion)
        erosion = cmath::soft_saturate(arrays->_water_velocity_map[idx], pars->erosion_rate, 1.0);
        break;
    }

    // apply layer erosiveness after the erosion calculation (seems best if we use soft_saturate)
    if (pars->_layer_mode_enabled) {
        // float layer_erosiveness = layer_erosiveness_array[exposed_layer];
        float layer_erosiveness = 1.0f; // ⚠️
        erosion *= layer_erosiveness;
    }

    erosion = cmath::clamp(erosion, 0.0f, available_erosion); // ensure not negative and not more than available_erosion

    // ================================================================
    // [Apply Erosion to Height]
    // ----------------------------------------------------------------
    if (pars->_layer_mode_enabled) {
        if (exposed_layer < pars->_layer_count) { // if layers not empty

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

        if (pars->_layer_mode_enabled) {
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

    if (pars->_layer_mode_enabled) {
        // already applied height to layer
    } else {
        height_map[idx] = cmath::clamp(height, pars->min_height, pars->max_height);
    }
    water_map[idx] = max(water, 0.0f);
    sediment_map[idx] = max(sediment, 0.0f);
}

#pragma endregion
// ================================================================================================================================
#pragma region MAIN

void TEMPLATE_CLASS_NAME::setup() {

    if (layer_map.is_valid() && !layer_map->empty()) { // layer mode
        _layer_mode_enabled = true;

        auto layer_map_shape = layer_map->shape();
        _layer_count = layer_map_shape[2];

        if (_layer_count > 8) throw std::runtime_error("_layer_count > 8");

        height_map.instantiate_if_null();
        height_map->resize(layer_map_shape[0], layer_map_shape[1]);

    } else if (height_map.is_valid() && !height_map->empty()) { // heightmap only
        _layer_mode_enabled = false;
        _layer_count = 1;

    } else {
        throw std::runtime_error("layermap or heightmap is not valid");
    }

    auto shape = height_map->shape();
    auto shape2 = std::array{shape[0], shape[1], (size_t)2};
    auto shape8 = std::array{shape[0], shape[1], (size_t)8};

    if (_layer_mode_enabled) ensure_array_ref_ready(_exposed_layer_map, shape); // need layer map if layer mode enabled

    set_par(_size, to_int2(shape));

    ensure_array_ref_ready(water_map, shape, true);
    ensure_array_ref_ready(sediment_map, shape, true);

    ensure_array_ref_ready(_water_out_map, shape);
    ensure_array_ref_ready(_sediment_out_map, shape);
    ensure_array_ref_ready(_water_velocity_map, shape);
    ensure_array_ref_ready(_slope_magnitude_map, shape);

    ensure_array_ref_ready(_slope_vector2_map, shape2);

    ensure_array_ref_ready(_flux8_map, shape8);
    ensure_array_ref_ready(_sediment_flux8_map, shape8);

    rain_map.instantiate_if_null(); // ensures we have a nullptr at least

    ready_device();
}
// --------------------------------------------------------------------------------------------------------------------------------
void TEMPLATE_CLASS_NAME::_compute() {

    setup();

    dim3 block(16, 16);
    dim3 grid((_size.x + block.x - 1) / block.x, (_size.y + block.y - 1) / block.y);

    for (int i = 0; i < steps; i++) {
        // ================================================================
        // [Kernels]
        // ----------------------------------------------------------------
        if (_layer_mode_enabled) {
            cmath::grid::layer_info_kernel<<<grid, block, 0, stream->get()>>>(
                _size,
                _layer_count,

                layer_map->dev_ptr(),         // in
                height_map->dev_ptr(),        // out
                _exposed_layer_map->dev_ptr() // out
            );
        }
        // ----------------------------------------------------------------
        cmath::grid::slope_vector_kernel<<<grid, block, 0, stream->get()>>>(
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
        // ----------------------------------------------------------------
        if (_pars.rain_rate > 0.0f) {
            add_rain3<<<grid, block, 0, stream->get()>>>(
                _size,
                water_map->dev_ptr(),
                rain_rate,
                rain_map->dev_ptr());
        }
        // ----------------------------------------------------------------
        calculate_flux3<<<grid, block, 0, stream->get()>>>(
            _dev_pars.dev_ptr(),
            _dev_arrays.dev_ptr(),
            _step);
        // ----------------------------------------------------------------
        apply_flux3<<<grid, block, 0, stream->get()>>>(
            _dev_pars.dev_ptr(),
            _dev_arrays.dev_ptr(),
            _step);
        // ================================================================

        _step++;
    }
}

#pragma endregion
// ================================================================================================================================

} // namespace TEMPLATE_NAMESPACE
