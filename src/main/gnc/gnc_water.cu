#include "gnc/gnc_water.cuh"

#include "core/cuda/math/grid.cuh"
#include "core/cuda/strings.cuh"
#include "core/defines.h"
#include "core/math/grid.h"

using core::strings::to_string;

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

// DH_CONST int2 GRID_OFFSETS_8[8] =
//     {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

DH_CONST int2 GRID_OFFSETS[8] =
    {
        {1, 0},
        {0, 1},
        {1, 1},
        {1, -1},
};

auto tiles_to_string(std::vector<int2> tiles) {
    return math::grid::tiles_to_string(tiles, "[ ]", " . ", true);
}




void TEMPLATE_CLASS_NAME::test() {

    printf("test()...\n");

    // for (int i = 1; i < 4; i++) {
    //     auto points = math::grid::square_ring(i);
    //     printf("%s\n", core::strings::to_string(points).c_str());
    // }

    auto offsets = math::grid::surrounding_offsets(3);

    // offsets = math::grid::filter_by_distance(offsets, 3.5f);
    printf("%s\n", to_string(offsets).c_str());
    printf(tiles_to_string(offsets).c_str());
    printf("count = %zu\n", offsets.size());

    printf("\n");

    offsets = math::grid::surrounding_offsets_half(3);
    printf("%s\n", to_string(offsets).c_str());
    printf(tiles_to_string(offsets).c_str());
    printf("count = %zu\n", offsets.size());
    printf("\n");

    offsets = math::grid::surrounding_offsets_quart(3);
    printf("%s\n", to_string(offsets).c_str());
    printf(tiles_to_string(offsets).c_str());
    printf("count = %zu\n", offsets.size());


    // // Array<float, 4> test_array = {1.0f, 1.0f, 1.0f, 1.0f};
    // Array<int2, 4> test_array = {
    //     int2{1, 0},
    //     int2{0, 1},
    //     int2{1, 1},
    //     int2{1, -1},
    // };
    // printf("%s\n", to_string(test_array.data).c_str());

    // core::cuda::DeviceArray<int2, 1> darray;
    // darray.upload(offsets);
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
