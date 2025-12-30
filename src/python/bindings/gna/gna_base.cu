#pragma once

#include "gna/gna_base.cuh"

namespace TEMPLATE_NAMESPACE {

// a kernel example makes a chequer pattern
__global__ void chequer_test(const size_t width, const size_t height, float *image_map) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    const int tile_size = 16;                       // probabally will be inlined (could also use constexpr)
    if ((x / tile_size + y / tile_size) % 2 == 0) { // for test make into a chequer pattern
        image_map[idx] = 0.0;
    }
}

void TEMPLATE_CLASS_NAME::process() {

#if GNA_CODE_ROUTE == 0

    ensure_arrays();
    auto &input_ref = *input;
    auto &output_ref = *output;

    if (input_ref.empty()) throw std::runtime_error("GNA_Base.input is empty");

    // output_ref.resize(input_ref.shape()); // ensure output is same shape
    output_ref = input_ref; // full copy, distinct buffer

    auto width = output_ref.width();
    auto height = output_ref.height();

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    chequer_test<<<grid, block, 0, stream.get()>>>(
        width, height,
        output_ref.dev_ptr());

    stream.sync();
#elif GNA_CODE_ROUTE == 1

    printf("🧠 GNA_CODE_ROUTE 1 TEST\n");

    if (input.is_valid()) {
        printf("🧠 input.is_valid()\n");
        printf("🧠 input->size() => %zu\n", input->size());
    }

    if (!input.is_valid()) throw std::runtime_error("GNA_Base.input is not valid");
    if (input->empty()) throw std::runtime_error("GNA_Base.input is empty");
    if (!output.is_valid()) output.instantiate();

    auto &input_ref = *input.shared_ptr;
    auto &output_ref = *output.shared_ptr;
    output_ref = input_ref; // should copy on gpu

    auto width = input->width();
    auto height = input->height();
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    chequer_test<<<grid, block, 0, stream.get()>>>(
        width, height,
        output->dev_ptr());

    stream.sync();


     printf("🧠 GNA_CODE_ROUTE 1 TEST FINISHED\n");

    // output.get() = input.get();
    // *output = *input;

#endif
}

} // namespace TEMPLATE_NAMESPACE
