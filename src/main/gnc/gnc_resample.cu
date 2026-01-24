#include "gnc/gnc_resample.cuh"
#include <stdexcept> // std::runtime_error

namespace TEMPLATE_NAMESPACE {

namespace cmath = core::cuda::math;

DH_INLINE int wrap_coord(float v, float max) {
    float m = fmodf(v, max);
    return (m < 0.0f) ? m + max : m;
}

DH_INLINE int clamp_coord(float v, float max) {
    return fminf(fmaxf(v, 0.0f), (max - 1));
}

DH_INLINE float sample_bilinear(const float *img,
                                int2 size,
                                float2 pos,
                                bool wrap) {

    float x = pos.x;
    float y = pos.y;

    // Apply addressing mode
    if (wrap) {
        x = wrap_coord(x, size.x);
        y = wrap_coord(y, size.y);
    } else {
        x = clamp_coord(x, size.x);
        y = clamp_coord(y, size.y);
    }

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = (x0 + 1) % size.x;
    int y1 = (y0 + 1) % size.y;

    float dx = x - x0;
    float dy = y - y0;

    float v00 = img[y0 * size.x + x0];
    float v10 = img[y0 * size.x + x1];
    float v01 = img[y1 * size.x + x0];
    float v11 = img[y1 * size.x + x1];

    return (1 - dx) * (1 - dy) * v00 +
           dx * (1 - dy) * v10 +
           (1 - dx) * dy * v01 +
           dx * dy * v11;
}

__global__ void resample_kernel(Parameters *pars,
                                const float *input, float *output,
                                int2 in_size,
                                int2 out_size,
                                const float *map_x, const float *map_y) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_size.x || y >= out_size.y)
        return;

    int idx = y * out_size.x + x;

    float src_x; // the x map (usually offset)
    float src_y; // the y map (usually offset)

    if (map_x) {
        src_x = map_x[idx];
    } else {
        src_x = 0.0; // assume 0 if no map
    }

    if (map_y) {
        src_y = map_x[idx];
    } else {
        src_y = 0.0; // assume 0 if no map
    }

    // scaling so that we're neutral to image resolution (0.5 would be half the output image width)
    if (pars->scale_by_output_size) {
        src_x *= out_size.x;
        src_y *= out_size.y;
    }

    // // OPTIONAL scaling
    // src_x *= pars->warp_x_strength;
    // src_y *= pars->warp_y_strength;

    // relative offset (most logical and easiest to feed map into)
    if (pars->relative_offset) {
        src_x += x;
        src_y += y;
    }

    output[idx] = sample_bilinear(input, in_size, make_float2(src_x, src_y), true);
}

// void TEMPLATE_CLASS_NAME::allocate_device() {
// }

// void TEMPLATE_CLASS_NAME::deallocate_device() {

//     // DeviceArray2D's
// #ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
// #define X(TYPE, NAME, DESCRIPTION) \
//     NAME.free_device();
//     TEMPLATE_CLASS_DEVICE_ARRAY_2DS
// #undef X
// #endif
// }

void TEMPLATE_CLASS_NAME::_compute() {

    if (!input.is_valid()) throw std::runtime_error("input is not valid");
    if (input->empty()) throw std::runtime_error("input is empty");

    auto shape = input->shape();

    _size = to_int2(shape);

    ensure_array_ref_ready(output, shape);

    ensure_array_ref_ready(map_x, shape, true); // fill 0's if not present
    ensure_array_ref_ready(map_y, shape, true); // fill 0's if not present

    dim3 block(16, 16);
    auto grid = cmath::calculate_grid(_size, block);

    ready_device();

    // // launch the kernel
    resample_kernel<<<grid, block, 0, stream->get()>>>(
        _dev_pars.dev_ptr(),
        input->dev_ptr(), output->dev_ptr(),
        _size,
        _size, // assuming same size output (unused feature here)
        map_x->dev_ptr(), map_y->dev_ptr());

    // // optional error check
    // auto err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    // }
}

} // namespace TEMPLATE_NAMESPACE
