#pragma once

#include "gna/gna_base.cuh"
#include "nanobind_helper.h"
// #include <string>
// #include <unordered_map>

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

// ⚠️ UNUSED??
template <typename T>
T get_attr_or_default(nb::handle self, const char *name, const T &default_value) {
    nb::dict d = nb::getattr(self, "__dict__");

    if (!d.contains(name))
        return default_value;

    try {
        return nb::cast<T>(d[name]);
    } catch (...) {
        return default_value;
    }
}

inline void bind(nb::module_ &m) {
    
    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

    // dynamic attributes (unused as it's hard to then get the vars from lower layers)
    // auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME), nb::dynamic_attr()).def(nb::init<>());

    // ================================================================================================================================

#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    ngd.def_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::NAME);
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    // ================================================================================================================================
    
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