/*
 */
#pragma once

#include "blur.cuh"
#include "python_helper.h"

namespace nb = nanobind;

inline void bind_blur(nb::module_ &m) {

    m.def("blur", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> arr, float amount, bool wrap) {
        int h = arr.shape(0);
        int w = arr.shape(1);
        blur::blur(arr.data(), w, h, amount, wrap); }, nb::arg("arr"), nb::arg("amount"), nb::arg("wrap") = false);
}