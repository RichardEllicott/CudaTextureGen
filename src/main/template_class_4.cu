#include "core/cuda/curand_array_2d.cuh"
#include "template_class_4.cuh"

#define TEMPLATE_DEMO_OPTION 1

namespace TEMPLATE_NAMESPACE {

// a kernel example makes a chequer pattern
__global__ void process_texture(const Parameters *pars, const size_t width, const size_t height, float *height_map) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    if (height_map) {
        const int tile_size = 16;                       // probabally will be inlined (could also use constexpr)
        if ((x / tile_size + y / tile_size) % 2 == 0) { // for test make into a chequer pattern
            height_map[idx] = 0.0;
        }
    }
}

// // kernel to make noise
// __global__ void generate_noise(const Parameters *pars, const size_t width, const size_t height, float *height_map, curandState *curand_states) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x >= width || y >= height)
//         return;

//     int idx = y * width + x;

//     curandState local_state = curand_states[idx];
//     float r = curand_uniform(&local_state); // [0,1)
//     height_map[idx] = r;
//     curand_states[idx] = local_state; // Save updated state

//     // height_map[idx] = 1.0; // TEST
// }

// 2D generate kernel
__global__ void generate_noise(float *out,
                               curandState *states,
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;

    curandState local = states[idx];
    float r = curand_uniform(&local); // (0,1]
    out[idx] = r;
    states[idx] = local; // save updated state
}

void TEMPLATE_CLASS_NAME::process() {

    image.upload();

    pars._width = image.get_width();
    pars._height = image.get_height();

    core::cuda::Stream stream;

    core::cuda::DeviceStruct<Parameters> _pars(pars); // automaticly uploads and free

    dim3 block(pars._block, pars._block);
    dim3 grid((pars._width + block.x - 1) / block.x,
              (pars._height + block.y - 1) / block.y);

#if TEMPLATE_DEMO_OPTION == 0

    process_texture<<<grid, block, 0, stream.get()>>>(
        _pars.dev_ptr(),
        pars.width, pars.height,
        image.dev_ptr());

#elif TEMPLATE_DEMO_OPTION == 1

    // auto curand_array = core::cuda::CurandArray();
    // curand_array.init(pars.width, pars.height, grid, block, stream.get());
    // stream.sync(); // important??

    auto curand_array = core::cuda::CurandArray2D(pars._width, pars._height, stream.get());
    stream.sync(); // important??

    generate_noise<<<grid, block, 0, stream.get()>>>(
        image.dev_ptr(),
        curand_array.dev_ptr(),
        pars._width, pars._height);

#endif

    stream.sync();

    image.download();
    image.free_device();
}

} // namespace TEMPLATE_NAMESPACE
