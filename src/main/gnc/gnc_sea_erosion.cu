#include "gnc/gnc_sea_erosion.cuh"

namespace TEMPLATE_NAMESPACE {

// 🚧 🚧 🚧 🚧 UNFINISHED
DH_INLINE float sea_fade_sine(float height, float avg_sea, float tide_range) {
    float half = 0.5f * tide_range;
    float low = avg_sea - half;
    float high = avg_sea + half;

    if (height <= low) return 1.0f;  // always submerged
    if (height >= high) return 0.0f; // never submerged

    float rel = (height - low) / tide_range;                  // 0..1
    float fade = 1.0f - acosf(1.0f - 2.0f * rel) / cmath::PI; // sine exposure fraction
    return fade;
}

__global__ void sea_pass3(
    const Parameters *__restrict__ pars,
    const ArrayPointers *__restrict__ arrays,
    const int step) {
    // ================================================================
    int2 map_size = pars->_size;
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    int idx = cmath::pos_to_idx(pos, map_size.x);
    // int idx8 = idx * 8;
    // ================================================================
    float *height_map = arrays->height_map;
    float *water_map = arrays->water_map;
    float *sediment_map = arrays->sediment_map;
    // // ================================================================
    // float height = height_map[idx];
    // float water = water_map[idx];
    // float sediment = sediment_map[idx];
    // // float surface = height + water;
    // // ================================================================
    // // [Tidal Fade]
    // // ----------------------------------------------------------------
    // float sea_fade = sea_fade_sine(height, pars->sea_level, pars->sea_tidal_range);
    // arrays->_sea_map[idx] = sea_fade;

    // // ================================================================
    // height_map[idx] = height;
    // water_map[idx] = water;
    // sediment_map[idx] = sediment;
}

void TEMPLATE_CLASS_NAME::_compute() {

    if (!height_map.is_valid()) throw std::runtime_error("height_map is not valid");
    if (height_map->empty()) throw std::runtime_error("height_map is empty");
    auto shape = height_map->shape();
    set_par(_size, make_int2(shape[0], shape[1]));
    // auto shape2 = std::array<size_t, 3>{shape[0], shape[1], 2};

    ensure_array_ref_ready(water_map, shape, true);
    ensure_array_ref_ready(sediment_map, shape, true);

    dim3 block(16, 16);
    auto grid = cmath::calculate_grid(_size, block);

    _ready_device();

    for (int i = 0; i < steps; i++) {

        sea_pass3<<<grid, block, 0, stream->get()>>>(
            _dev_pars.dev_ptr(),
            _dev_arrays.dev_ptr(),
            _step);

        _step++;
    }
}

} // namespace TEMPLATE_NAMESPACE
