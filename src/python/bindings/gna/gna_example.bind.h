#pragma once

#include "gna/gna_example.cuh"
#include "nanobind_helper.h"

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    // init standard object
    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

    // ================================================================================================================================

    // bind Ref<DeviceArray>'s
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    ngd.def_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::NAME);
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    // ================================================================================================================================

    // dynamic properties fallback to the has/get/set storage
    nb::helper::bind_dynamic_properties<TEMPLATE_CLASS_NAME>(
        ngd,
        &TEMPLATE_CLASS_NAME::has_property,
        &TEMPLATE_CLASS_NAME::get_property,
        &TEMPLATE_CLASS_NAME::set_property);

    // ================================================================================================================================

    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) {
        self.process();
    });
}

} // namespace TEMPLATE_NAMESPACE