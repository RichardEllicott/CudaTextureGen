#pragma once

#include "gna/gna_base.cuh"
#include "nanobind_helper.h"
#include <string>
#include <unordered_map>

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    ngd.def_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::NAME);
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    ngd.def("__getattr__", [](TEMPLATE_CLASS_NAME &self, const char *key) { return self.getattr(std::string(key)); });
    ngd.def("__setattr__", [](TEMPLATE_CLASS_NAME &self, const char *key, nb::object value) { self.setattr(std::string(key), value); });

    ngd.def("process", [](TEMPLATE_CLASS_NAME &self) { self.process(); });
}

} // namespace TEMPLATE_NAMESPACE