#include "core/cuda/math.cuh"
#include "gnc/gnc_erosion.cuh"

namespace TEMPLATE_NAMESPACE {

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

// a kernel example makes a chequer pattern
__global__ void calculate_layer_height(
    const int width, const int height, const int layers,
    const float *__restrict__ layermap, // in
    float *__restrict__ heightmap       // out
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;
    int layer_idx = idx * layers;

    // float layer_height = 0.0;
    // for (int n = 0; n < layers; ++n) {
    //     layer_height += layermap[layer_idx + n];
    // }

    // heightmap[idx] = layer_height;

    heightmap[idx] = get_layered_height(layermap, layers, layer_idx);
}

void TEMPLATE_CLASS_NAME::process() {

    if (layermap.is_valid() && !layermap->empty()) { // layer mode
        _layer_mode = true;
        heightmap.instantiate_if_null();
        auto shape = layermap->shape();
        heightmap->resize(shape[0], shape[1]);
        _layers = shape[2];

    } else if (heightmap.is_valid() && !heightmap->empty()) { // heightmap only
        _layer_mode = false;
        _layers = 1;

    } else {
        throw std::runtime_error("layermap or heightmap is not valid");
    }

    width = heightmap->width();
    height = heightmap->height();

    stream.instantiate_if_null();

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // calculate the layer height for layer mode
    if (_layer_mode) {
        calculate_layer_height<<<grid, block, 0, stream->get()>>>(
            width, height, _layers,
            layermap->dev_ptr(),
            heightmap->dev_ptr());
    }

    // output.instantiate_if_null();           // if no DeviceArray make one
    // *output.shared_ptr = *input.shared_ptr; // will copy the memory (on the gpu) from input to output (by dereferencing)
}

} // namespace TEMPLATE_NAMESPACE
