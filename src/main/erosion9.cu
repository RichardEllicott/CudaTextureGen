#include "core/cuda/curand_array_2d.cuh"
#include "erosion9.cuh"
#include "erosion9_kernels.cuh"
// #include "noise_util.cuh"
#include "core.h" // timer

namespace TEMPLATE_NAMESPACE {

#pragma region KERNELS

// Accessors
#define DEFINE_MAP_ACCESSORS(NAME)                                                                     \
    __device__ inline float get_##NAME##_map(const ArrayPtrs *arrays, int step, int idx) {             \
        return ((step % 2 == 0) ? arrays->NAME##_map : arrays->_##NAME##_map_out)[idx];                \
    }                                                                                                  \
    __device__ inline void set_##NAME##_map(const ArrayPtrs *arrays, int step, int idx, float value) { \
        ((step % 2 == 0) ? arrays->_##NAME##_map_out : arrays->NAME##_map)[idx] = value;               \
    }
DEFINE_MAP_ACCESSORS(height)
DEFINE_MAP_ACCESSORS(water)
DEFINE_MAP_ACCESSORS(sediment)
#undef DEFINE_MAP_ACCESSORS

// new pattern
__global__ void calculate_flux2(
    const Parameters *pars,
    const ArrayPtrs *arrays,
    const int step) {
    // ================================================================
    int width = pars->_width;
    int height = pars->_height;
    // ----------------------------------------------------------------
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;
    // ================================================================
    float height = get_height_map(arrays, step, idx);
    float water = get_water_map(arrays, step, idx);
    float sediment = get_sediment_map(arrays, step, idx);

    //
    //
    //
    //

    set_height_map(arrays, step, idx, height);
    set_water_map(arrays, step, idx, water);
    set_sediment_map(arrays, step, idx, sediment);
}

#pragma endregion

// example allocation
void TEMPLATE_CLASS_NAME::allocate_device() {
    if (_device_allocated)
        return;

    pars._width = height_map.dimensions()[0];
    pars._height = height_map.dimensions()[1];
    size_t array_size = pars._width * pars._height;

// allocate and zero arrays
#define ALLOCATE_ARRAYS \
    X(water_map)        \
    X(sediment_map)
#define X(NAME)                                   \
    if (NAME.size() != array_size) {              \
        NAME.resize({pars._width, pars._height}); \
        NAME.zero_device();                       \
    }
    ALLOCATE_ARRAYS
#undef X
#undef ALLOCATE_ARRAYS

// allocate arrays
#define ALLOCATE_ARRAYS  \
    X(_height_map_out)   \
    X(_water_map_out)    \
    X(_sediment_map_out) \
    X(_slope_map)
#define X(NAME)                                   \
    if (NAME.size() != array_size) {              \
        NAME.resize({pars._width, pars._height}); \
    }
    ALLOCATE_ARRAYS
#undef X
#undef ALLOCATE_ARRAYS

    // flux output
    _flux8.resize({array_size * 8});
    _sediment_flux8.resize({array_size * 8});

    _device_allocated = true;
}

void TEMPLATE_CLASS_NAME::process00() {

    allocate_device();
    configure_device();
    stream.sync();

    core::util::Timer timer;

    for (int step = 0; step < pars.steps; ++step) {

        // if we have rain
        if (pars.rain_rate > 0.0f) {
            rain_pass<<<grid, block, 0, stream.get()>>>(
                dev_pars.dev_ptr(), pars._width, pars._height,

                nullptr,

                rain_map.dev_ptr(), // optional in

                water_map.dev_ptr() // out
            );
        }

        calculate_flux<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(), pars._width, pars._height,
            step,

            height_map.dev_ptr(),   // in
            water_map.dev_ptr(),    // in
            sediment_map.dev_ptr(), // in

            _flux8.dev_ptr(),          // out
            _sediment_flux8.dev_ptr(), // out
            _slope_map.dev_ptr()       // out

        );

        apply_flux<<<grid, block, 0, stream.get()>>>(
            dev_pars.dev_ptr(), pars._width, pars._height,

            height_map.dev_ptr(),      // in
            water_map.dev_ptr(),       // in
            sediment_map.dev_ptr(),    // in
            _flux8.dev_ptr(),          // in
            _sediment_flux8.dev_ptr(), // in
            _slope_map.dev_ptr(),      // in

            hardness_map.dev_ptr(), // optional in

            _height_map_out.dev_ptr(),  // out
            _water_map_out.dev_ptr(),   // out
            _sediment_map_out.dev_ptr() // out
        );

        // flip the in/out maps
        std::swap(height_map, _height_map_out);
        std::swap(water_map, _water_map_out);
        std::swap(sediment_map, _sediment_map_out);
    }

    timer.mark_time();
    pars._calculation_time = timer.elapsed_seconds();
    if (pars.debug_print) {
        printf("⏱️ calculation time: %.3f seconds\n", timer.elapsed_seconds());
        // core::logging::printf("⏱️ calculation time: %.3f seconds\n", timer.elapsed_seconds());
    }
}

void TEMPLATE_CLASS_NAME::process01() {

    allocate_device();
    configure_device();
    stream.sync();

    core::cuda::DeviceStruct<ArrayPtrs> dev_array_ptrs(get_array_ptrs()); // device side pars

    for (int step = 0; step < pars.steps; ++step) {
    }
}

void TEMPLATE_CLASS_NAME::process() {

    switch (pars.mode) {
    case 0:
        process00();
        break;
    case 1:
        process01();
        break;
    case 2:
        break;
    case 3:
        break;
    }
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
