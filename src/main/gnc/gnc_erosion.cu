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







void TEMPLATE_CLASS_NAME::process() {

    // testing seed generator
    printf("seed_test = %u\n", CONSTEXPR_LINE_SEED);
    printf("seed_test = %u\n", CONSTEXPR_LINE_SEED);
    printf("seed_test = %u\n", CONSTEXPR_LINE_SEED);

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
    width = height_map->width();
    height = height_map->height();

    _exposed_layer_map.instantiate_if_null();
    _exposed_layer_map->resize(width, height);

    _slope_vector2_map.instantiate_if_null();
    _slope_vector2_map->resize(width, height, 2);

    water_map.instantiate_if_null();
    if (water_map->shape() != height_map_shape) {
        water_map->resize(height_map_shape);
        water_map->zero_device();
    }

    stream.instantiate_if_null();

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    int2 map_size = {width, height};

    // calculate the layer height for layer mode
    if (_layer_mode) {
        layer_calculations<<<grid, block, 0, stream->get()>>>(
            width, height, _layer_count,
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
        jitter,
        _step,
        true,
        0, // jitter mode
        1.0f, //scale
        CONSTEXPR_LINE_SEED);

    _step++;

    // output.instantiate_if_null();           // if no DeviceArray make one
    // *output.shared_ptr = *input.shared_ptr; // will copy the memory (on the gpu) from input to output (by dereferencing)
}

} // namespace TEMPLATE_NAMESPACE
