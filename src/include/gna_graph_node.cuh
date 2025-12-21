/*

GN-A

🦄 testing new idea of node more directly controlled by python

*/
#pragma once

#include "core/cuda/device_array.cuh"
#include "macros.h"
#include <memory> // shared_ptr

#define TEMPLATE_CLASS_NAME GNA_GraphNode
#define TEMPLATE_NAMESPACE gna_graph_node

namespace TEMPLATE_NAMESPACE {

class TEMPLATE_CLASS_NAME {

  public:
    std::shared_ptr<core::cuda::DeviceArray<float, 2>> output;

    void process() {
        printf("process...\n");

        if (output) {
            printf("output...\n");
            printf("%zu\n", output->size());
            printf("\n");
        }
    }
};

} // namespace TEMPLATE_NAMESPACE
