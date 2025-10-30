#include "core.h"
#include "erosion2.cuh"
#include "simple_erode.cuh"
#include <curand_kernel.h>

// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>

namespace TEMPLATE_NAMESPACE {

void TEMPLATE_CLASS_NAME::process() {

    core::CudaStream stream; // create a stream

    height_map.upload();
    pars.width = height_map.get_width();
    pars.height = height_map.get_height();

    sediment_map.resize(pars.width, pars.height);
    sediment_map.clear(); // ensure zeros (i think required)
    sediment_map.upload();

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    auto timer = core::Timer();

    for (int s = 0; s < pars.steps; ++s) {
        simple_erode<<<grid, block, 0, stream.get()>>>(pars.width, pars.height,
                                                       height_map.dev_ptr(),
                                                       sediment_map.dev_ptr(),
                                                       nullptr,
                                                       pars.wrap,
                                                       pars.jitter,
                                                       pars.erosion_rate,
                                                       pars.slope_threshold,
                                                       pars.deposition_rate);
    }

    stream.sync(); // sync the stream

    timer.mark_time();
    printf("calculation time: %.2f ms\n", timer.elapsed_seconds());

    height_map.download();
    height_map.free_device();

    sediment_map.download();
    sediment_map.free_device();
}

} // namespace TEMPLATE_NAMESPACE
