#include "gnc/gnc_normal_map.cuh"

#include "core/cuda/math.cuh"

namespace TEMPLATE_NAMESPACE {

__global__ void normal_map_kernel(
    const float *__restrict__ heightmap, // in
    float *__restrict__ normalmap,       // out 3x bigger

    int2 map_size,

    float normal_scale, bool wrap,
    bool direct_x_style = true) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    int idx = cmath::pos_to_idx(pos, map_size);
    // ================================================================

    // Offsets clockwise from top
    constexpr int2 offsets8[8] = {{0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}};

    float samples[8];
    for (int i = 0; i < 8; ++i) {
        int2 sample_pos = pos + offsets8[i];
        sample_pos = cmath::wrap_or_clamp_index(sample_pos, map_size, wrap);
        int sample_idx = cmath::pos_to_idx(sample_pos, map_size);
        samples[i] = heightmap[sample_idx];
    }

    float2 d = make_float2(
        (samples[1] + 2.0f * samples[2] + samples[3]) -
            (samples[7] + 2.0f * samples[6] + samples[5]),

        (samples[5] + 2.0f * samples[4] + samples[3]) -
            (samples[7] + 2.0f * samples[0] + samples[1]));

    float3 n = make_float3(
        d.x * normal_scale,
        -d.y * normal_scale,
        1.0f);

    n = cmath::normalize(n);

    // Convert normal to [0,1] color space
    float3 c = make_float3(
        0.5f + 0.5f * n.x,
        0.5f + 0.5f * (direct_x_style ? -n.y : n.y),
        0.5f + 0.5f * n.z);

    // Store
    int idx3 = idx * 3;
    normalmap[idx3 + 0] = c.x;
    normalmap[idx3 + 1] = c.y;
    normalmap[idx3 + 2] = c.z;
}

void TEMPLATE_CLASS_NAME::_compute() {

    if (!input.is_valid()) throw std::runtime_error("input is not valid"); // if no DeviceArray, error
    if (input->empty()) throw std::runtime_error("input is empty");        // if DeviceArray is empty, error

    auto shape = input->shape();
    _size = to_int2(shape);
    auto shape3 = std::array{shape[0], shape[1], (size_t)3}; // RGB size
    ensure_array_ref_ready(output, shape3);

    dim3 block(16, 16);
    dim3 grid = cmath::calculate_grid(_size, block);

    // ready_device();

    normal_map_kernel<<<grid, block, 0, stream->get()>>>(
        input->dev_ptr(),
        output->dev_ptr(),
        _size,
        normal_scale,
        direct_x_style);
}

} // namespace TEMPLATE_NAMESPACE
