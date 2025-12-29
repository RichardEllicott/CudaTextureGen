#pragma once

#include "gna/gna_base.cuh"
#include "nanobind_helper.h"
#include <string>
#include <unordered_map>

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    // OLD
    // auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

    // allow dynamic attributes on the object (allows setting class pars)
    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME), nb::dynamic_attr()).def(nb::init<>());

    // ================================================================================================================================

#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    ngd.def_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::NAME);
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif



    // ================================================================================================================================

    // was trying to use unordered map

    // ngd.def("__getattr__", [](TEMPLATE_CLASS_NAME &self, const char *key) { return self.getattr(std::string(key)); });
    // ngd.def("__setattr__", [](TEMPLATE_CLASS_NAME &self, const char *key, nb::object value) { self.setattr(std::string(key), value); });

    // // version will not override existing
    // ngd.def("__setattr__", [](nb::handle self, const char *key, nb::object value) {
    //     // Try normal setattr first
    //     if (nb::hasattr(self, key)) {
    //         nb::setattr(self, key, value);
    //         return;
    //     }

    //     // Otherwise fallback to your dynamic storage
    //     nb::cast<TEMPLATE_CLASS_NAME &>(self).setattr(key, value);
    // });

    // ================================================================================================================================

    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) { self.process(); });
}

} // namespace TEMPLATE_NAMESPACE