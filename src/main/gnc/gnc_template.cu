#include "gnc/gnc_template.cuh"

namespace TEMPLATE_NAMESPACE {

// a kernel example makes a chequer pattern
__global__ void chequer_test(
    const int width, const int height,
    float *image_map,
    const int tile_size = 16) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;

    if ((x / tile_size + y / tile_size) % 2 == 0) { // for test make into a chequer pattern
        image_map[idx] = 0.0;
    }
}

void TEMPLATE_CLASS_NAME::_compute() {

    if (!input.is_valid()) throw std::runtime_error("GNA_Base.input is not valid"); // if no DeviceArray, error
    if (input->empty()) throw std::runtime_error("GNA_Base.input is empty");        // if DeviceArray is empty, error

    output.instantiate_if_null();           // if no DeviceArray make one
    *output.shared_ptr = *input.shared_ptr; // will copy the memory (on the gpu) from input to output (by dereferencing)

    width = input->width();
    height = input->height();
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    stream.instantiate_if_null();

    chequer_test<<<grid, block, 0, stream->get()>>>(
        width, height,
        output->dev_ptr(),
        tile_size);

    // // stream->sync();
}

void TEMPLATE_CLASS_NAME::test() {
}

} // namespace TEMPLATE_NAMESPACE
