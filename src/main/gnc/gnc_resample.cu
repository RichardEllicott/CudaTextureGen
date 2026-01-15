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
                                int width, int height,
                                float x, float y,
                                bool wrap) {

    // Apply addressing mode
    if (wrap) {
        x = wrap_coord(x, width);
        y = wrap_coord(y, height);
    } else {
        x = clamp_coord(x, width);
        y = clamp_coord(y, height);
    }

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = (x0 + 1) % width;
    int y1 = (y0 + 1) % height;

    float dx = x - x0;
    float dy = y - y0;

    float v00 = img[y0 * width + x0];
    float v10 = img[y0 * width + x1];
    float v01 = img[y1 * width + x0];
    float v11 = img[y1 * width + x1];

    return (1 - dx) * (1 - dy) * v00 +
           dx * (1 - dy) * v10 +
           (1 - dx) * dy * v01 +
           dx * dy * v11;
}

__global__ void resample_kernel(Parameters *pars,
                                const float *input, float *output,
                                int in_w, int in_h,
                                int out_w, int out_h,
                                const float *map_x, const float *map_y) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h)
        return;

    int idx = y * out_w + x;

    if (pars->mode == 0) {

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
            src_x *= out_w;
            src_y *= out_h;
        }

        // OPTIONAL scaling
        src_x *= pars->warp_x_strength;
        src_y *= pars->warp_y_strength;

        // relative offset (most logical and easiest to feed map into)
        if (pars->relative_offset) {
            src_x += x;
            src_y += y;
        }

        output[idx] = sample_bilinear(input, in_w, in_h, src_x, src_y, true);

    } else if (pars->mode == 1) {

        // --- plain rotation around center ---
        float angle = pars->angle; // note radians

        float cosA = cosf(angle);
        float sinA = sinf(angle);

        // center of output image
        float cx = 0.5f * out_w;
        float cy = 0.5f * out_h;

        // coordinates relative to center
        float dx = (float)x - cx;
        float dy = (float)y - cy;

        // apply inverse rotation (output -> input mapping)
        float src_x = cosA * dx + sinA * dy + cx;
        float src_y = -sinA * dx + cosA * dy + cy;

        // --- apply normalized offset ---
        // offset_x = 0.5 means half the width, offset_y = 0.5 means half the height
        float offset_px = pars->offset_x * out_w;
        float offset_py = pars->offset_y * out_h;
        src_x += offset_px;
        src_y += offset_py;

        // sample from input
        output[idx] = sample_bilinear(input, in_w, in_h, src_x, src_y, true);
    }
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

    if (input.is_null() || input->empty())
        throw std::runtime_error("input is null or empty");

    auto input_shape = input->shape();

    _size = to_int2(input_shape);

    ensure_array_ref_ready(output, input_shape);

    ensure_array_ref_ready(map_x, input_shape, true); // fill 0's if not present
    ensure_array_ref_ready(map_y, input_shape, true); // fill 0's if not present

    ready_device();

    dim3 block(16, 16);
    dim3 grid((_size.x + block.x - 1) / block.x,
              (_size.y + block.y - 1) / block.y);

    // // launch the kernel
    resample_kernel<<<grid, block, 0, stream->get()>>>(
        dev_pars.dev_ptr(),
        input->dev_ptr(), output->dev_ptr(),
        _size.x, _size.y,
        _size.x, _size.y, // assuming same size output (unused feature here)
        map_x->dev_ptr(), map_y->dev_ptr());

    // // optional error check
    // auto err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    // }
}

} // namespace TEMPLATE_NAMESPACE
