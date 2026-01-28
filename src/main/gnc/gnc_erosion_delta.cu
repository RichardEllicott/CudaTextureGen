#include "gnc/gnc_erosion_delta.cuh"

#include "core/cuda/grid.cuh"
#include "core/cuda/math/grid.cuh"

namespace TEMPLATE_NAMESPACE {

__global__ void calculate_flux4(
    const Parameters *__restrict__ pars,
    const ArrayPointers *__restrict__ arrays,
    const int step) {
    // ================================================================================================================================
    int2 map_size = pars->_size;
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= map_size.x || pos.y >= map_size.y) return;
    // ================================================================================================================================
    int idx = cmath::pos_to_idx(pos, map_size.x);
    int idx2 = idx * 2;
    // ================================================================================================================================
    float *height_map = arrays->height_map;
    float *water_map = arrays->water_map;
    float *sediment_map = arrays->sediment_map;
    // ================================================================================================================================
}

void STAGE_CALCULATE_SLOPE_VECTORS(TEMPLATE_CLASS_NAME &self){


    // cmath::grid::slope_vector_kernel<<<self.grid, self.block, 0, stream->get()>>>(
    //         _size,

    //         height_map->dev_ptr(), // in
    //         water_map->dev_ptr(),  // in
    //         nullptr,
    //         _slope_vector2_map->dev_ptr(), // out

    //         wrap,

    //         0, // jitter mode
    //         slope_jitter,
    //         _step,
    //         0x865C34F3u,

    //         1.0f,
    //         _slope_magnitude_map->dev_ptr());

// }


}

void TEMPLATE_CLASS_NAME::test() {

    printf("test123()...\n");

    auto sfg = core::cuda::grid::SamplingFieldGenerator();
    sfg.print_test_data();
}

void TEMPLATE_CLASS_NAME::_compute() {



// cmath::grid::slope_vector_kernel<<<grid, block, 0, stream->get()>>>(
//             _size,

//             height_map->dev_ptr(), // in
//             water_map->dev_ptr(),  // in
//             nullptr,
//             _slope_vector2_map->dev_ptr(), // out

//             wrap,

//             0, // jitter mode
//             slope_jitter,
//             _step,
//             0x865C34F3u,

//             1.0f,
//             _slope_magnitude_map->dev_ptr());

}

} // namespace TEMPLATE_NAMESPACE
