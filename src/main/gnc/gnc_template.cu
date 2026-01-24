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

    if (!input.is_valid()) throw std::runtime_error("input is not valid");
    if (input->empty()) throw std::runtime_error("input is empty");
    auto shape = input->shape();
    set_par(_size, make_int2(shape[0], shape[1]));
    auto shape2 = std::array<size_t, 3>{shape[0], shape[1], 2};

    output.instantiate_if_null();           // if no DeviceArray make one
    *output.shared_ptr = *input.shared_ptr; // will copy the memory (on the gpu) from input to output (by dereferencing)

    dim3 block(16, 16);
    auto grid = cmath::calculate_grid(_size, block);

    ready_device();

    chequer_test<<<grid, block, 0, stream->get()>>>(
        _size.x, _size.y,
        output->dev_ptr(),
        tile_size);
}

void TEMPLATE_CLASS_NAME::test() {
    printf("test()...\n");
}

} // namespace TEMPLATE_NAMESPACE
