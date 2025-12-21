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

    // // bind pars
    // #ifdef TEMPLATE_CLASS_PARAMETERS
    // #define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
//     ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::get_##NAME, &TEMPLATE_CLASS_NAME::set_##NAME, DESCRIPTION);
    //     TEMPLATE_CLASS_PARAMETERS
    // #undef X
    // #endif

    //     // bind DeviceArray's
    // #ifdef TEMPLATE_CLASS_DEVICE_ARRAY_NS
    // #define X(TYPE, DIMENSIONS, NAME, DESCRIPTION)                                                                                                         \
//     auto get_##NAME = [](TEMPLATE_CLASS_NAME &self) { return nb::helper::numpy::to_array(self.NAME); };                                                \
//     auto set_##NAME = [](TEMPLATE_CLASS_NAME &self, nb::ndarray<float, nb::c_contig> array) { nb::helper::numpy::to_device_array(array, self.NAME); }; \
//     ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME, DESCRIPTION);
    //     TEMPLATE_CLASS_DEVICE_ARRAY_NS
    // #undef X
    // #endif

    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) {
        self.process();
    });
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
