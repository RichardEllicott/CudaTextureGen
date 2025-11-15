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

void TEMPLATE_CLASS_NAME::process() {

}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
