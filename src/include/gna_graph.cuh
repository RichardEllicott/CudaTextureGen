/*

GN-A

🦄 testing new idea of node more directly controlled by python

┌─────┬───────┐
│ Col1│ Col2  │
├─────┼───────┤
│ A   │ B     │
└─────┴───────┘
╔═════╦═══════╗
║ Col1║ Col2  ║
╠═════╬═══════╣
║ A   ║ B     ║
╚═════╩═══════╝

*/
#pragma once

#include "core/cuda/types/device_array.cuh"
#include "core/macros.h"
#include <memory> // shared_ptr
#include <variant>

// ================================================================================================================================
// [Definitions]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNA_Graph
#define TEMPLATE_NAMESPACE gna_graph
// --------------------------------------------------------------------------------------------------------------------------------
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS                             \
    X(size_t, _width, 1024, "map width")                      \
    X(size_t, _height, 1024, "map height")                    \
    X(size_t, _block, 16, "block size (best to leave at 16)") \
    X(bool, test_bool, 0.0, "a test bool")                    \
    X(float, test_float, 0.0, "a test float")                 \
    X(int, test_int, 0.0, "a test int")
// --------------------------------------------------------------------------------------------------------------------------------
// (TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_DEVICE_ARRAYS                                                   \
    X(float, 2, 1, height_map, "current height, set this map to start the simulation") \
    X(float, 2, 1, water_map, "current water, optionally set this map at start")       \
    X(float, 2, 1, _water_out, "current water, optionally set this map at start")
// ================================================================================================================================

namespace TEMPLATE_NAMESPACE {

// using Variant = std::variant<int, float, std::string>; // my variant type




class TEMPLATE_CLASS_NAME {

  public:
    std::shared_ptr<core::cuda::types::DeviceArray<float, 2>> output;

    // std::unordered_map<std::string, Variant> props;

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
