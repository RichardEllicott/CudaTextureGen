#include "core/cuda/curand_array_2d.cuh"
#include "template_darray_1.cuh"

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

// example allocation
void TEMPLATE_CLASS_NAME::allocate_device() {
    if (device_allocated)
        return;

    pars._width = image.width();
    pars._height = image.height();

    size_t array_size = pars._width * pars._height;

// allocate DeviceArray2D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION)              \
    if (NAME.empty()) {                         \
        NAME.resize(pars._width, pars._height); \
        NAME.zero_device();                     \
    }
    TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif

// allocate DeviceArray's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSION, NAME, DESCRIPTION) \
    NAME.resize(array_size *DIMENSION);       \
    NAME.zero_device();
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    device_allocated = true;
}





void TEMPLATE_CLASS_NAME::process() {

    allocate_device();  // allocate memory
    configure_device(); // run before launching a kernel

    process_texture<<<grid, block, 0, stream.get()>>>(dev_pars.dev_ptr(), pars._width, pars._height, image.dev_ptr());
}

void TEMPLATE_CLASS_NAME::test_process() {
}

void TEMPLATE_CLASS_NAME::test_process2() {
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
