#include "grid.cuh"

namespace core::cuda::math::grid {

#pragma region SLOPES

__global__ void slope_vector_kernel(
    const int2 map_size,

    const float *__restrict__ height_map1, // in
    const float *__restrict__ height_map2, // in (optional)
    const float *__restrict__ height_map3, // in (optional)

    float *__restrict__ slope_vector2_map, // out (map is a double size float map, storing vectors pairs)

    const bool wrap,

    const int jitter_mode,
    const float jitter,
    const int step,
    const int jitter_seed,

    const float scale,

    float *__restrict__ slope_magnitude  // optional slope magnitude map
) {
    // ================================================================
    int2 pos = global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    int idx = pos_to_idx(pos, map_size);
    int idx2 = idx * 2;
    // ================================================================

    float2 slope_vector2 = compute_slope_vector(
        height_map1,
        height_map2,
        height_map3,

        map_size,
        pos,

        wrap,

        jitter_mode,
        jitter,
        step,
        jitter_seed,

        scale);

    // slope_vector2_map[idx2] = slope_vector2.x;
    // slope_vector2_map[idx2 + 1] = slope_vector2.y;
    store_float2(slope_vector2_map, idx2, slope_vector2);

    if (slope_magnitude){
        slope_magnitude[idx] = length(slope_vector2);
    }

}

// syntax example only, will crash with null maps
static void example_slope_vector() {

    int2 map_size = {128, 128};

    dim3 block(16, 16);
    dim3 grid((map_size.x + block.x - 1) / block.x, (map_size.y + block.y - 1) / block.y);
    cudaStream_t stream = nullptr;

    float *height_map1 = nullptr;
    float *height_map2 = nullptr;
    float *height_map3 = nullptr;
    float *slope_vector2_map = nullptr;

    bool wrap = true;
    int jitter_mode = 0;
    float jitter = 0.0f;
    int step = 0;
    int jitter_seed = 0x7BA3BE90u;

    float scale = 1.0f;

    // launch example
    slope_vector_kernel<<<grid, block, 0, stream>>>(
        map_size,

        height_map1,
        height_map2,
        height_map3,
        slope_vector2_map,

        wrap,

        jitter_mode,
        jitter,
        step,
        jitter_seed,

        scale);
}

#pragma endregion

#pragma region LAYERS

__global__ void layer_info_kernel(
    const int2 size,
    const int layer_count,

    const float *__restrict__ layer_map, // in
    float *__restrict__ height_map,      // out
    int *__restrict__ _exposed_layer     // out

) {
    // ================================================================
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size.x || y >= size.y) return;
    int idx = y * size.x + x;
    int layer_idx = idx * layer_count;
    // ================================================================
    const float *pixel_layers = layer_map + layer_idx; // pointer to layer itself
    LayerInfo info = compute_layer_info(layer_count, pixel_layers);
    // ================================================================
    height_map[idx] = info.total_height;
    _exposed_layer[idx] = info.exposed_layer;
}

#pragma endregion

} // namespace core::cuda::math::grid
