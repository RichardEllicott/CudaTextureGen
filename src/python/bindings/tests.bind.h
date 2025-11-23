#pragma once

#include "nanobind_helper.h"
#include "tests.cuh"

#include "core/cuda/device_array_n.cuh"

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

    m.def("test_device_array_n", []() {
        printf("test_device_array_n...");


        core::cuda::DeviceArrayN<float, 2> device_array_n;

    });
}
} // namespace tests