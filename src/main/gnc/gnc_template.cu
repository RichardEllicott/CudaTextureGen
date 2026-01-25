#include "gnc/gnc_template.cuh"

namespace TEMPLATE_NAMESPACE {

// a kernel example makes a chequer pattern
__global__ void chequer_test(
    int2 size,
    float *image_map,
    const int tile_size = 16) {

    int2 pos = cmath::global_thread_pos2();

    if (pos.x >= size.x || pos.y >= size.y) return;
    int idx = cmath::pos_to_idx(pos, size);

    if ((pos.x / tile_size + pos.y / tile_size) % 2 == 0) { // for test make into a chequer pattern
        image_map[idx] = 0.0;
    }
}

void TEMPLATE_CLASS_NAME::_compute() {

    if (!input.is_valid()) throw std::runtime_error("GNA_Base.input is not valid"); // if no DeviceArray, error
    if (input->empty()) throw std::runtime_error("GNA_Base.input is empty");        // if DeviceArray is empty, error

    output.instantiate_if_null();           // if no DeviceArray make one
    *output.shared_ptr = *input.shared_ptr; // will copy the memory (on the gpu) from input to output (by dereferencing)

    auto shape = input->shape();
    set_par(_size, to_int2(shape));

    dim3 block(16, 16);
    auto grid = cmath::calculate_grid(_size, block);

    stream.instantiate_if_null();

    chequer_test<<<grid, block, 0, stream->get()>>>(
        _size,
        output->dev_ptr(),
        tile_size);
}

void TEMPLATE_CLASS_NAME::test() {
    printf("test()...\n");
}

} // namespace TEMPLATE_NAMESPACE
