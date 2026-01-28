#include "gnc/gnc_water.cuh"

#include "core/cuda/math/grid.cuh"

namespace TEMPLATE_NAMESPACE {

__global__ void velocity_cell_water(
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

    // build _water_lateral_velocity based on slopes
}

void TEMPLATE_CLASS_NAME::test() {

    printf("test()...\n");
}

void TEMPLATE_CLASS_NAME::_compute() {

    if (!height_map.is_valid()) throw std::runtime_error("height_map is not valid");
    if (height_map->empty()) throw std::runtime_error("height_map is empty");
    auto shape = height_map->shape();
    set_par(_size, make_int2(shape[0], shape[1]));
    auto shape2 = std::array<size_t, 3>{shape[0], shape[1], 2};

    ensure_array_ref_ready(water_map, shape, true);
    ensure_array_ref_ready(sediment_map, shape, true);

    ensure_array_ref_ready(_water_vertical_velocity, shape, true);
    ensure_array_ref_ready(_water_lateral_velocity, shape2, true);

    dim3 block(16, 16);
    auto grid = cmath::calculate_grid(_size, block);

    _ready_device();

    for (int i = 0; i < steps; i++) {

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

            1.0f);

        // sea_pass3<<<grid, block, 0, stream->get()>>>(
        //     _dev_pars.dev_ptr(),
        //     _dev_arrays.dev_ptr(),
        //     _step);

        _step++;
    }
}

} // namespace TEMPLATE_NAMESPACE
