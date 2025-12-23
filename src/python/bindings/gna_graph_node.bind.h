/*



*/
#pragma once

#include "GNA_graph_node.cuh"
#include "nanobind_helper.h"

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, "GraphNode").def(nb::init<>());

    ngd.def_rw("output", &TEMPLATE_CLASS_NAME::output);

    // bind with par names
    ngd.def(
        "connect_input",
        &TEMPLATE_CLASS_NAME::connect_input,
        nb::arg("output_node"),
        nb::arg("output_port"),
        nb::arg("input_port"));

    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) {
        self.process();
    });
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
