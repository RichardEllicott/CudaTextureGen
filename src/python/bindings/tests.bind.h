#pragma once

#include "python_helper.h"
#include "tests.cuh"

namespace tests {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {

    // auto ngd = nb::class_<Tests>(m, "Tests").def(nb::init<>());

    m.def("print_debug_info", []() {
        tests::print_debug_info();
    });

    m.def("cuda_hello", []() {
        tests::cuda_hello();
    });
}
} // namespace tests