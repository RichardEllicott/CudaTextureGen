#include "core/cuda/math.cuh"
#include "core/cuda/math/grid.cuh"
#include "core/math.h"
#include "gnc/gnc_erosion.cuh"

#define PRECALCULATE_EXPOSED_LAYER

namespace TEMPLATE_NAMESPACE {

namespace math = core::math;
namespace cmath = core::cuda::math;

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

__global__ void calculate_outflow3(
    const Parameters *__restrict__ pars,
    const ArrayPointers *__restrict__ arrays,
    // DebugOutputs *__restrict__ debug,
    const int step) {
    // ================================================================

    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    int idx = cmath::pos_to_idx(pos, map_size);
    int idx2 = idx * 2;

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

    float2 slope_vector = {arrays->_slope_vector2_map[idx2], arrays->_slope_vector2_map[idx2 + 1]};
    float slope_magnitude = cmath::length(slope_vector);

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

__global__ void apply_flux3(
    const Parameters *__restrict__ pars,
    const ArrayPointers *__restrict__ arrays,
    // DebugOutputs *__restrict__ debug,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = cmath::pos_to_idx(pos, map_size.x);
    int idx2 = idx * 2;
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
    bool layer_mode = pars->_layer_count > 0; // if layer mode active
    int exposed_layer;
    int layer_idx;
    if (layer_mode) {
        layer_idx = idx * pars->_layer_count;

#ifdef PRECALCULATE_EXPOSED_LAYER
        exposed_layer = arrays->_exposed_layer_map[idx];
#else
        exposed_layer = get_exposed_layer(arrays->layer_map, pars->_layer_count, layer_idx);
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

    float2 slope_vector = {arrays->_slope_vector2_map[idx2], arrays->_slope_vector2_map[idx2 + 1]};
    float slope_magnitude = cmath::length(slope_vector);

    switch (pars->erosion_mode) {
    case 0: // simple water * erosion_rate
        erosion = water_out * pars->erosion_rate;
        break;
    case 1:
        erosion = water_out * pars->erosion_rate * slope_magnitude; // with slope_magnitude ⚠️ BROKE?
        break;
    case 2:
        erosion = water * pars->erosion_rate * slope_magnitude; // maybe based on total water?
        break;
    case 3:
        erosion = water * pars->erosion_rate * arrays->_water_velocity_map[idx]; // total water and the water velocity (manning)
        break;
    case 4: // soft saturation scheme (limits the max erosion)
        erosion = cmath::soft_saturate(arrays->_water_velocity_map[idx], pars->erosion_rate, 1.0);
        break;
    }

    // apply layer erosiveness after the erosion calculation (seems best if we use soft_saturate)
    if (layer_mode) {
        float layer_erosiveness = arrays->layer_erosiveness_array[exposed_layer];
        erosion *= layer_erosiveness;
    }

    erosion = cmath::clamp(erosion, 0.0f, available_erosion); // ensure not negative and not more than available_erosion

    // ================================================================
    // [Apply Erosion to Height]
    // ----------------------------------------------------------------
    if (layer_mode) {
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
        height_map[idx] = cmath::clamp(height, pars->min_height, pars->max_height);
    }
    water_map[idx] = max(water, 0.0f);
    sediment_map[idx] = max(sediment, 0.0f);
}

void TEMPLATE_CLASS_NAME::setup() {
}

void TEMPLATE_CLASS_NAME::_compute() {

    // ================================================================
    // [Setup]
    // ----------------------------------------------------------------

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

#define CODE_ROUTE 0 // route 0 has restrictions but is failing linux?
#if CODE_ROUTE == 0
    ensure_array_ref_ready(water_map, height_map_shape, true);
    ensure_array_ref_ready(sediment_map, height_map_shape, true);

    ensure_array_ref_ready(_water_out_map, height_map_shape);
    ensure_array_ref_ready(_sediment_out_map, height_map_shape);

    ensure_array_ref_ready(_flux8_map, std::array{_width, _height, (size_t)8});
    ensure_array_ref_ready(_sediment_flux8_map, std::array{_width, _height, (size_t)8});
    ensure_array_ref_ready(_slope_vector2_map, std::array{_width, _height, (size_t)2});

    // ensure_array_ref_ready(_slope_magnitude_map, height_map_shape);
    ensure_array_ref_ready(_water_velocity_map, height_map_shape);
    rain_map.instantiate_if_null(); // ensures we have a nullptr

    // ensure_array_ref_ready(sediment_layer_map, std::array{_width, _height, (size_t)2});

    ensure_array_ref_ready(_exposed_layer_map, height_map_shape);
    ensure_array_ref_ready(_sea_map, height_map_shape);

    ensure_array_ref_ready(_wind_vector2_map, std::array{_width, _height, (size_t)2});

#elif CODE_ROUTE == 1
#endif
#undef CODE_ROUTE

    stream.instantiate_if_null();

    dim3 block(16, 16);
    dim3 grid((_width + block.x - 1) / block.x, (_height + block.y - 1) / block.y);

    int2 map_size = {(int)_width, (int)_height};

    ready_device();

    // ================================================================
    // [Kernels]
    // ----------------------------------------------------------------

    for (int i = 0; i < steps; i++) {

        // calculate the layer height for layer mode
        if (_layer_mode) {
            cmath::grid::layer_info_kernel<<<grid, block, 0, stream->get()>>>(
                _width, _height, _layer_count,
                layer_map->dev_ptr(),         // in
                height_map->dev_ptr(),        // out
                _exposed_layer_map->dev_ptr() // out
            );
        }

        cmath::grid::slope_vector_kernel<<<grid, block, 0, stream->get()>>>(
            map_size,

            height_map->dev_ptr(), // in
            water_map->dev_ptr(),  // in
            nullptr,
            _slope_vector2_map->dev_ptr(), // out

            pars.wrap,

            0, // jitter mode
            slope_jitter,
            _step,
            0x865C34F3u,

            1.0f

        );

        calculate_outflow3<<<grid, block, 0, stream->get()>>>(
            dev_pars.dev_ptr(),
            dev_array_pointers.dev_ptr(),
            _step);

        apply_flux3<<<grid, block, 0, stream->get()>>>(
            dev_pars.dev_ptr(),
            dev_array_pointers.dev_ptr(),
            _step);

        _step++;
    }

    // output.instantiate_if_null();           // if no DeviceArray make one
    // *output.shared_ptr = *input.shared_ptr; // will copy the memory (on the gpu) from input to output (by dereferencing)
}

} // namespace TEMPLATE_NAMESPACE
