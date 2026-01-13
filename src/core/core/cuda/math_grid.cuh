/*

computer slope vector's kernel

exposed layers etc

global kernel code in .cu file due to one-definition-rule

*/
#pragma once

#include "math.cuh"

namespace core::cuda::math::grid {

#pragma region SLOPES

// compute slope vectors, with up to 3 maps, optional jitter
DH_INLINE float2 compute_slope_vector(
    const float *__restrict__ height_map1, // in
    const float *__restrict__ height_map2, // in (optional, set as nullptr if not required)
    const float *__restrict__ height_map3, // in (optional, set as nullptr if not required)

    const int2 map_size,
    const int2 pos,

    const bool wrap = true,

    const int jitter_mode = 0,
    const float jitter = 0.0f,
    const int step = 0,
    const int jitter_seed = 0x549C952Bu,

    const float scale = 1.0f) {

    int xp = wrap_or_clamp_index(pos.x + 1, map_size.x, wrap); // x + 1
    int xn = wrap_or_clamp_index(pos.x - 1, map_size.x, wrap); // x - 1
    int yp = wrap_or_clamp_index(pos.y + 1, map_size.y, wrap); // y + 1
    int yn = wrap_or_clamp_index(pos.y - 1, map_size.y, wrap); // y - 1

    int xp_idx = pos.y * map_size.x + xp; // {+1,0}
    int xn_idx = pos.y * map_size.x + xn; // {-1,0}
    int yp_idx = yp * map_size.x + pos.x; // {0,+1}
    int yn_idx = yn * map_size.x + pos.x; // {0,-1}

    float xp_height = height_map1[xp_idx]; // x+ height
    float yp_height = height_map1[yp_idx]; // y+ height
    float xn_height = height_map1[xn_idx]; // x- height
    float yn_height = height_map1[yn_idx]; // y- height

    if (height_map2) {
        xp_height += height_map2[xp_idx];
        yp_height += height_map2[yp_idx];
        xn_height += height_map2[xn_idx];
        yn_height += height_map2[yn_idx];
    }
    if (height_map3) {
        xp_height += height_map3[xp_idx];
        yp_height += height_map3[yp_idx];
        xn_height += height_map3[xn_idx];
        yn_height += height_map3[yn_idx];
    }

    // // scale
    xp_height /= scale;
    yp_height /= scale;
    xn_height /= scale;
    yn_height /= scale;

    // ================================================================
    // [Jitter]
    // ----------------------------------------------------------------
    if (jitter > 0.0f) {
        switch (jitter_mode) {
        case 0: { // cheaper, reuses one hash, lower quality random shouldn't be a problem over frames
            uint32_t h = hash_uint(pos.x, pos.y, step, jitter_seed);
            xp_height += hash_to_4randf(h, 0) * jitter;
            yp_height += hash_to_4randf(h, 1) * jitter;
            xn_height += hash_to_4randf(h, 2) * jitter;
            yn_height += hash_to_4randf(h, 3) * jitter;
            break;
        }
        case 1: { // uses 4 hashes, technically better random
            xp_height += hash_float_signed(pos.x, pos.y, step, jitter_seed + 0) * jitter;
            yp_height += hash_float_signed(pos.x, pos.y, step, jitter_seed + 1) * jitter;
            xn_height += hash_float_signed(pos.x, pos.y, step, jitter_seed + 2) * jitter;
            yn_height += hash_float_signed(pos.x, pos.y, step, jitter_seed + 3) * jitter;
            break;
        }
        }
    }

    return float2{xn_height - xp_height, yn_height - yp_height};
}

DH_INLINE void store_float2(float *base, int idx, float2 v) {
    float *p = base + idx;
    p[0] = v.x;
    p[1] = v.y;
}

// #ifdef __CUDACC__

__global__ void slope_vector_kernel(
    const int2 map_size,

    const float *__restrict__ height_map1, // in
    const float *__restrict__ height_map2, // in (optional)
    const float *__restrict__ height_map3, // in (optional)

    float *__restrict__ slope_vector2_map, // out (map is a double size float map, storing vectors pairs)

    const bool wrap = true,

    const int jitter_mode = 0,
    const float jitter = 0.0f,
    const int step = 0,
    const int jitter_seed = 0xCCA39754u,

    const float scale = 1.0f);

// #endif

#pragma endregion

#pragma region LAYERS

struct LayerInfo {
    float total_height;
    int exposed_layer; // -1 is none
};

// find the layer total height and exposed layer
D_INLINE LayerInfo compute_layer_info(
    int layer_count,
    const float *__restrict__ layer_map // pointer to position in layer map
) {
    // ================================================================
    LayerInfo layer_info = {0.0f, -1};

    for (int n = layer_count - 1; n >= 0; --n) {
        float value = layer_map[n];
        layer_info.total_height += value;

        if (layer_info.exposed_layer == -1 && value > 0.0f) layer_info.exposed_layer = n; // found exposed layer
    }
    // ================================================================
    return layer_info;
}

// found in cu file
__global__ void layer_info_kernel(
    const int width, const int height, const int layer_count,
    const float *__restrict__ layer_map, // in
    float *__restrict__ height_map,      // out
    int *__restrict__ _exposed_layer     // out

);

#pragma endregion

} // namespace core::cuda::math
