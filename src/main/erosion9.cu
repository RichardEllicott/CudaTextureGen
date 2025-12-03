#include "core/cuda/curand_array_2d.cuh"
#include "erosion9.cuh"
#include "erosion9_kernels.cuh"
// #include "noise_util.cuh"
#include "core.h" // timer
#include "cuda_math.cuh"

namespace TEMPLATE_NAMESPACE {

#pragma region KERNELS

// Atomic erosion primitive
template <typename T>
struct MinFilter {
    __device__ T operator()(T a, T b) const {
        return a < b ? a : b;
    }
};

template <typename T>
struct Clamp {
    __device__ T operator()(T val, T minv, T maxv) const {
        return max(minv, min(val, maxv));
    }
};

// Kernel template that applies a sequence of stages
template <typename T, typename... Stages>
__global__ void erosionKernel(T *data, int n, Stages... stages) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = data[idx];
        // Apply each stage in sequence
        (void)std::initializer_list<int>{
            ((val = stages(val)), 0)... // fold expression trick
        };
        data[idx] = val;
    }
}

// // Compose erosion pipeline at compile time
// erosionKernel<<<grid, block>>>(d_data, n,
//     MinFilter<float>{},
//     Clamp<float>{});

// ping pong helper
template <typename MapPtr>
__device__ inline float read_map_in(MapPtr in, MapPtr out, int step, int idx) {
    return (step % 2 == 0 ? in : out)[idx];
}
// ping pong helper
template <typename MapPtr>
__device__ inline void write_map_out(MapPtr in, MapPtr out, int step, int idx, float value) {
    (step % 2 == 0 ? out : in)[idx] = value;
}

template <typename T>
__device__ inline T *get_map_ptr(T *in, T *out, int step) {
    return (step % 2 == 0 ? in : out);
}

__device__ inline int pos_to_idx(int2 pos, int map_width) {
    return pos.y * map_width + pos.x;
}

// calulate total height of layers
__global__ void calc_layer_height(
    const Parameters *pars,
    const ArrayPtrs *arrays,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = pos_to_idx(pos, map_size.x);
    // ----------------------------------------------------------------

    auto layer_map = get_map_ptr(arrays->layer_map, arrays->_layer_map_out, step);
    auto height_map = get_map_ptr(arrays->height_map, arrays->_height_map_out, step);
    auto water_map = get_map_ptr(arrays->water_map, arrays->_water_map_out, step);
    auto sediment_map = get_map_ptr(arrays->sediment_map, arrays->_sediment_map_out, step);

    // auto layer_map_out = get_map_ptr(arrays->_layer_map_out, arrays->layer_map, step);
    // auto height_map_out = get_map_ptr(arrays->_height_map_out, arrays->height_map, step);
    // auto water_map_out = get_map_ptr(arrays->_water_map_out, arrays->water_map, step);
    // auto sediment_map_out = get_map_ptr(arrays->_sediment_map_out, arrays->sediment_map, step);

    int layer_count = pars->_layers;
    int layer_idx = idx * layer_count;

    float height = 0.0;
    for (int i = 0; i < layer_count; i++) {
        height += layer_map[layer_idx + i];
    }

    float water = water_map[idx];

    float surface = height + water;

    height_map[idx] = height;

    arrays->_surface_map[idx] = surface;
}

// new pattern
__global__ void calculate_flux2(
    const Parameters *pars,
    const ArrayPtrs *arrays,
    const int step) {
    // ================================================================
    int2 map_size = make_int2(pars->_width, pars->_height);
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= map_size.x || pos.y >= map_size.y) // bounds check
        return;
    int idx = pos_to_idx(pos, map_size.x);
    // ----------------------------------------------------------------
    int mode = pars->mode;
    // ================================================================
    float height = read_map_in(arrays->height_map, arrays->_height_map_out, step, idx);
    float water = read_map_in(arrays->water_map, arrays->_water_map_out, step, idx);
    float sediment = read_map_in(arrays->sediment_map, arrays->_sediment_map_out, step, idx);
    // ================================================================
    // [Calculate Slopes]
    // ----------------------------------------------------------------

    float surface;
    int layer_count;
    float *layers_ptr;

    switch (mode) {
    case 0:
        surface = height + water;
        break;
    case 1:
        layer_count = pars->_layers;
        layers_ptr = arrays->layer_map + idx * layer_count;

        // if (pars->_layers > 1) {
        // }

        break;
    }

    float slopes[8];
    for (int n = 0; n < 8; ++n) {
        int2 new_pos = wrap_or_clamp_index(pos + offsets[n], map_size, pars->wrap);
        int new_idx = pos_to_idx(new_pos, map_size.x);
        float new_height = read_map_in(arrays->height_map, arrays->_height_map_out, step, new_idx);
        float new_water = read_map_in(arrays->height_map, arrays->_height_map_out, step, new_idx);
        float new_sediment = read_map_in(arrays->height_map, arrays->_height_map_out, step, new_idx);
    }

    // ================================================================
    // won't do this here
    write_map_out(arrays->height_map, arrays->_height_map_out, step, idx, height);
    write_map_out(arrays->water_map, arrays->_water_map_out, step, idx, water);
    write_map_out(arrays->sediment_map, arrays->_sediment_map_out, step, idx, sediment);
}

#pragma endregion

#pragma region ARRAY_WORKER

// Generic layout converter between SoA and AoS for 3D arrays
// direction = true  → SoA → AoS
// direction = false → AoS → SoA
template <typename T>
inline void convert_array_layout(const T *source, T *destination, int width, int height, int channels, bool soa_to_aos) {
    if (channels == 1) { // Just copy, no rearrangement needed
        std::memcpy(destination, source, sizeof(T) * width * height);
        return;
    }
    int plane_size = width * height;
    for (int idx = 0; idx < plane_size; ++idx) {
        for (int c = 0; c < channels; ++c) {
            if (soa_to_aos) {
                destination[idx * channels + c] = source[c * plane_size + idx];
            } else {
                destination[c * plane_size + idx] = source[idx * channels + c];
            }
        }
    }
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
        calculate_flux2<<<grid, block, 0, stream.get()>>>(dev_pars.dev_ptr(), dev_array_ptrs.dev_ptr(), step);
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
