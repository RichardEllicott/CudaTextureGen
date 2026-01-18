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
    //
    //
    // float *height_map = arrays->height_map;
    // float *water_map = arrays->water_map;
    // float *sediment_map = arrays->sediment_map;
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
}

} // namespace TEMPLATE_NAMESPACE
