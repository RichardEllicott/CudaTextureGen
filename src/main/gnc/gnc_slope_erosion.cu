#include "gnc/gnc_slope_erosion.cuh"

namespace TEMPLATE_NAMESPACE {

// working basic erode, has no water just sediment redistribution
// tested with ping pong but makes no difference!!
__global__ void simple_erode(
    Parameters *pars,
    const int width, const int height, const int step,
    const float *heightmap,
    const float *sediment,
    float *heightmap_out,
    float *sediment_out // seems to make no difference using ping pong

) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    float h = heightmap[idx];
    float s = sediment[idx];

    // 8-way neighbor offsets
    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    float total_slope = 0.0f;
    float slopes[8] = {0};

    // Compute slopes to neighbors
    for (int i = 0; i < 8; ++i) {

        int nx;
        int ny;

        if (pars->wrap) {
            nx = (x + dx[i] + width) % width;
            ny = (y + dy[i] + height) % height;
        } else {
            nx = x + dx[i];
            ny = y + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;
        }

        int nIdx = ny * width + nx;
        float nh = heightmap[nIdx];
        float slope = h - nh; // amount higher than this neighbour

        if (pars->jitter > 0.0f) {
            uint32_t hash = cmath::hash_uint(x, y, step, 0x54130ED9u);
            slope += cmath::hash_float_signed(hash) * pars->jitter;
        }

        if (slope > pars->slope_threshold) {
            slopes[i] = slope;
            total_slope += slope;
        }
    }

    // Erode and deposit based on slope
    float eroded = pars->erosion_rate * total_slope;
    h -= eroded;
    s += eroded;

    // Distribute sediment to neighbors
    for (int i = 0; i < 8; ++i) {
        if (slopes[i] > 0) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;

            int nIdx = ny * width + nx;
            float share = (slopes[i] / total_slope) * pars->deposition_rate * s;

            // Atomic to avoid race conditions
            atomicAdd(&heightmap_out[nIdx], share);
            atomicAdd(&sediment_out[nIdx], -share);
        }
    }

    // Write back
    heightmap_out[idx] = h;
    sediment_out[idx] = s;
}

void TEMPLATE_CLASS_NAME::_compute() {

    if (height_map.is_null() || height_map->empty())
        throw std::runtime_error("height_map is null or empty");

    auto height_map_shape = height_map->shape();
    _size = to_int2(height_map_shape);

    ensure_array_ref_ready(sediment_map, height_map_shape, true);

    dim3 block(16, 16);
    dim3 grid((_size.x + block.x - 1) / block.x, (_size.y + block.y - 1) / block.y);

    ready_device();

    for (int i = 0; i < steps; i++) {

        simple_erode<<<grid, block, 0, stream->get()>>>(
            _dev_pars.dev_ptr(),
            _size.x, _size.y, _step,
            height_map->dev_ptr(),
            sediment_map->dev_ptr(),
            height_map->dev_ptr(),
            sediment_map->dev_ptr());

        _step++;
    }

    // pars._siz
}

} // namespace TEMPLATE_NAMESPACE
