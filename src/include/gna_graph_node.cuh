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

#include "core/cuda/device_array.cuh"
#include "macros.h"
#include <memory> // shared_ptr
#include <variant>
#include "template_macro_undef.h"


// ================================================================================================================================
// [Definitions]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNA_GraphNode
#define TEMPLATE_NAMESPACE gna_graph_node
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

using Variant = std::variant<int, float, std::string>; // my variant type

// gna graph node
class TEMPLATE_CLASS_NAME {

  public:
    std::shared_ptr<core::cuda::DeviceArray<float, 2>> output;
    std::unordered_map<std::string, Variant> props;

    // ================================================================================================================================
    // []
    // --------------------------------------------------------------------------------------------------------------------------------
    std::string name = "<NAME>";
    bool debug = true;
    // ================================================================================================================================
    // [Connections]
    // --------------------------------------------------------------------------------------------------------------------------------

    struct Connection {
        size_t port;                             // the output port index on the upstream node
        std::weak_ptr<TEMPLATE_CLASS_NAME> node; // the upstream node
    };

    // std::vector<std::weak_ptr<TEMPLATE_CLASS_NAME>> inputs; // we store the connections going backwards
    std::vector<Connection> inputs; // we store the connections going backwards

    void _cyclic_check() {

        for (size_t i = 0; i < inputs.size(); ++i) {

            auto start = inputs[i].node.lock();
            if (!start) {
                inputs[i].node.reset(); // if dead ref, clear this input
                continue;
            }

            std::unordered_set<const TEMPLATE_CLASS_NAME *> visited;
            std::vector<const TEMPLATE_CLASS_NAME *> to_check;

            to_check.push_back(start.get());
            visited.insert(this); // mark self so we detect returning to it

            bool cycle_found = false;

            while (!to_check.empty()) {

                const auto check = to_check.back();
                to_check.pop_back();

                if (visited.count(check)) {
                    cycle_found = true;
                    break;
                }

                visited.insert(check);

                for (auto &c : check->inputs) {
                    if (auto s = c.node.lock()) {
                        to_check.push_back(s.get());
                    }
                }
            }

            if (cycle_found) {
                inputs[i].node.reset(); // if cycle detected, clear this input
            }
        }
    }




    // Connect input from node output
bool connect_input(
    const std::shared_ptr<TEMPLATE_CLASS_NAME> &output_node,
    const size_t output_port, // output (from upstream node)
    const size_t input_port   // input port
) {
    if (input_port >= inputs.size()) {
        throw std::out_of_range("connect_input: no such input index");
    }

    if (debug) {
        std::cout << "[connect_input] "
                  << "this=" << name
                  << " input_port=" << input_port
                  << " ← upstream=" << (output_node ? output_node->name : "<null>")
                  << " output_port=" << output_port
                  << std::endl;
    }

    // Assign the connection
    inputs[input_port].node = output_node;
    inputs[input_port].port = output_port;

    // Run cycle detection
    _cyclic_check();

    const bool ok = !inputs[input_port].node.expired();

    if (debug) {
        if (ok) {
            std::cout << "[connect_input] connection accepted for " << name
                      << " input_port=" << input_port << std::endl;
        } else {
            std::cout << "[connect_input] connection rejected (cycle) for " << name
                      << " input_port=" << input_port << std::endl;
        }
    }

    return ok;
}


    // LIKELY A BAD IDEA
    // bool connect_output(
    //     size_t output_port,
    //     const std::shared_ptr<TEMPLATE_CLASS_NAME> &input_target,
    //     size_t input_port) {

    //     if (!input_target) {
    //         throw std::invalid_argument("connect_output: target node is null");
    //     }

    //     // Delegate to the target node’s input connector
    //     return input_target->connect_input(
    //         input_port,         // which input on the target
    //         shared_from_this(), // this node (upstream)
    //         output_port         // which output port on this node
    //     );
    // }

    void process() {
        printf("process...\n");

        if (output) {
            printf("output...\n");
            printf("%zu\n", output->size());
            printf("\n");
        }
    }

    TEMPLATE_CLASS_NAME() {
        inputs.resize(4);
    }
};

} // namespace TEMPLATE_NAMESPACE
