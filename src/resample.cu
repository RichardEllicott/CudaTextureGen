#include "resample.cuh"

namespace resample {

__device__ __forceinline__ int wrap_coord(float v, float max) {
    float m = fmodf(v, max);
    return (m < 0.0f) ? m + max : m;
}

__device__ __forceinline__ int clamp_coord(float v, float max) {
    return fminf(fmaxf(v, 0.0f), (max - 1));
}

__device__ float sample_bilinear(const float *img,
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

#define RESAMPLE_RELATIVE

__global__ void resample_kernel(const float *input, float *output,
                                int in_w, int in_h,
                                int out_w, int out_h,
                                const float *map_x, const float *map_y) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h)
        return;

    int idx = y * out_w + x;

#ifdef RESAMPLE_RELATIVE
    // Treat map_x/map_y as relative offsets from (x, y)
    float src_x = x + map_x[idx];
    float src_y = y + map_y[idx];
#elif
    // map_x/map_y give the input coords for this output pixel
    float src_x = map_x[idx];
    float src_y = map_y[idx];

#endif

    output[idx] = sample_bilinear(input, in_w, in_h, src_x, src_y, true);
}

void Resample::process() {

    core::CudaStream stream; // create a stream

    size_t width = input.get_width();
    size_t height = input.get_width();
    size_t _block = 16;

    input.upload();
    map_x.upload();
    map_y.upload();

    output.resize(input.get_width(), input.get_height());
    output.allocate_device(); // allocate memory (no upload)

    dim3 block(_block, _block);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    resample_kernel<<<grid, block, 0, stream.get()>>>(
        input.device_ptr, output.device_ptr,
        width, height, width, height, map_x.device_ptr, map_y.device_ptr);

    output.download();

    input.free_device();
    output.free_device();
    map_x.free_device();
    map_y.free_device();

    stream.sync();
}

} // namespace resample