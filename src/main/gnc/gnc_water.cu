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

#define DH_CONST __device__ __constant__ const

// DH_CONST int2 GRID_OFFSETS_8[8] =
//     {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

DH_CONST int2 GRID_OFFSETS_8[8] =
    {
        {1, 0},
        {0, 1},
        {1, 1},
        {1, -1},
};

#undef DH_CONST

// quater a square perimeter ring
std::vector<int2> square_ring_quart(int radius = 1) {
    std::vector<int2> out;
    int l = radius * 2;
    for (int i = 0; i < l; i++) {
        int2 pos = make_int2(-radius + 1 + i, radius); // gives the seg from (-n+1,n) → (n,n)
        out.push_back(pos);
    }
    return out;
}

// half a square perimeter ring
std::vector<int2> square_ring_half(int radius = 1) {
    std::vector<int2> out;
    int l = radius * 2;

    // original quarter: top edge from (-n+1, n) → (n, n)
    for (int i = 0; i < l; i++) {
        int2 p = make_int2(-radius + 1 + i, radius);
        out.push_back(p);

        // rotated 90° CCW: (x, y) → (-y, x)
        int2 r = make_int2(-p.y, p.x);
        out.push_back(r);
    }

    return out;
}

void print_int2_list(const std::vector<int2> &pts) {
    std::cout << "{";
    for (size_t i = 0; i < pts.size(); ++i) {
        const auto &p = pts[i];
        std::cout << "{" << p.x << "," << p.y << "}";
        if (i + 1 < pts.size())
            std::cout << ", ";
    }
    std::cout << "}\n";
}




void TEMPLATE_CLASS_NAME::test() {

    printf("test()...\n");

    for (int i = 1; i < 4; i++) {

        auto points = square_ring_quart(i);

        print_int2_list(points);
    }

    /*




    */

    // cmath::GRID_OFFSETS_8
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
